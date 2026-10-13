"""失败驱动重检索（Failure-driven Retrieval，任务6）。

把 fix_b.py 里针对 11 道计算题的硬编码修复逻辑，抽象成通用三段式：

    First Pass  →  failure detection  →  targeted retrieval  →  re-answer

关键原则（避免"伪造置信度"）：
  * 不使用任何无来源的 confidence score。
  * 失败判定只用**可观测、可复现**的信号：
      - empty_answer            答案抽取失败
      - format_violation_*      答案不符合题目要求的格式（缺 %、小数位不符、非数字…）
      - numeric_evidence_miss   题目中的强数字/条款跨度在检索证据中完全没出现
      - self_consistency_disagreement  多次采样结果不一致（仅当开启多采样）
  * 每条信号都写进日志，便于人工复核。
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config

from .context_budget import ContextBudget
from .memory_compressor import EvidenceChunk
from .pipeline import DynamicMemoryPipeline, PipelineResult, append_jsonl
from .text_features import QuestionFeatures, Span, extract_question_features, span_coverage

# 公司/产品 → 年报 doc_id 映射（原先硬编码在 fix_b.py，这里作为单一数据源）
COMPANY_DOC_MAP: Dict[str, List[str]] = {
    "比亚迪": ["annual_byd_2024_report", "annual_byd_2025_report"],
    "宁德时代": ["annual_catl_2024_report", "annual_catl_2025_report"],
    "美的集团": ["annual_midea_2024_report", "annual_midea_2025_report"],
    "招商银行": ["annual_cmb_2025_report"],
    "中国建筑": ["annual_cscec_2024_report", "annual_cscec_2025_report"],
    "中国移动": ["annual_chinamobile_2025_report"],
}


# ---------------------------------------------------------------- 失败判定

@dataclass
class FailureSignal:
    is_failure: bool
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"is_failure": self.is_failure, "reasons": list(self.reasons),
                "details": dict(self.details)}


_PERCENT_HINT = re.compile(r"百分[数号]|％|%|百分比")
_DECIMAL_HINT = re.compile(r"保留\s*([0-9一二三四五六两])\s*位小数")
_RATIO_HINT = re.compile(r"(率|占比|比例|增速|增幅|跌幅|收益率)")
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6}
NUMERIC_KINDS = ("number", "money", "percent", "date", "clause")


def _decimal_count(s: str) -> int:
    m = re.search(r"\d+\.(\d+)", s)
    return len(m.group(1)) if m else 0


def detect_failure(question: str, answer: str,
                   q_type: Optional[str] = None,
                   answer_format: Optional[str] = None,
                   option_keys: Sequence[str] = (),
                   evidence_text: str = "",
                   key_spans: Sequence[Span] = (),
                   candidate_answers: Sequence[str] = (),
                   first_pass_complete: bool = True) -> FailureSignal:
    """可观测的失败信号判定（不产生任何"猜出来的置信度"）。"""
    reasons: List[str] = []
    details: Dict[str, object] = {}
    tpe = q_type or ""
    fmt = (answer_format or "").lower()
    is_choice = bool(option_keys) or fmt in ("mcq", "multi", "tf")
    is_calc = tpe in ("计算题", "抽取题") or (not option_keys and not is_choice)

    ans = (answer or "").strip()
    if not ans:
        reasons.append("empty_answer")

    if is_choice:
        letters = set("".join(re.findall(r"[A-Z]+", ans.upper())))
        allowed = {k.upper() for k in option_keys} or set("ABCD")
        if letters and not letters <= allowed:
            reasons.append("format_violation_choice")
            details["invalid_letters"] = sorted(letters - allowed)

    if is_calc and ans:
        if not re.search(r"\d", ans):
            reasons.append("format_violation_numeric")
        # 百分号要求：题目明确提到百分数/%，或题目在问"率/占比/比例"且要求保留小数位
        needs_percent = bool(_PERCENT_HINT.search(question)) or bool(
            _RATIO_HINT.search(question) and _DECIMAL_HINT.search(question))
        if needs_percent and "%" not in ans and "％" not in ans:
            reasons.append("format_violation_percent")
        m = _DECIMAL_HINT.search(question)
        if m and re.search(r"\d", ans):
            wanted = _CN_NUM.get(m.group(1), None)
            if wanted is None:
                try:
                    wanted = int(m.group(1))
                except ValueError:
                    wanted = None
            if wanted is not None and _decimal_count(ans) != wanted:
                reasons.append("format_violation_decimals")
                details["wanted_decimals"] = wanted
        if len(ans) > 40:
            reasons.append("answer_too_long_for_numeric")

    # 数字证据缺失：题目里的**数值型**强跨度（金额/百分比/条款号/多位数字）在证据里一个都没出现
    strong = [s for s in key_spans if s.is_strong]
    numeric_strong = [s for s in strong if s.kind in NUMERIC_KINDS]
    if numeric_strong and evidence_text:
        cov = span_coverage(numeric_strong, evidence_text)
        details["numeric_strong_total"] = cov["strong_total"]
        details["numeric_strong_hit"] = cov["strong_hit"]
        if cov["strong_hit"] == 0 and len(numeric_strong) >= 2:
            reasons.append("numeric_evidence_miss")
        elif cov["strong_total"] and cov["strong_hit"] / max(1, cov["strong_total"]) < 0.34:
            reasons.append("numeric_evidence_low_coverage")

    if candidate_answers and len({c for c in candidate_answers if c}) >= 2:
        reasons.append("self_consistency_disagreement")
        details["candidates"] = list(candidate_answers)

    if not first_pass_complete:
        reasons.append("first_pass_incomplete")

    return FailureSignal(is_failure=bool(reasons), reasons=reasons, details=details)


# ---------------------------------------------------------------- 定向重检索

def targeted_doc_ids(features: QuestionFeatures, domain: str,
                     base_doc_ids: Sequence[str] = ()) -> List[str]:
    """根据题目实体（公司/产品名）选择重检索目标文档。

    计算题最常见失败原因是"正确年报没被召回"，因此优先用实体匹配锁定文档；
    匹配不到时回退到原本的候选文档集合。
    """
    hit: List[str] = []
    for name in features.entity_names:
        for company, docs in COMPANY_DOC_MAP.items():
            if company in name or name in company:
                hit.extend(docs)
    if hit:
        # 去重保序
        return list(dict.fromkeys(hit))
    return list(base_doc_ids)


@dataclass
class RetryRecord:
    qid: str = ""
    tier: str = ""
    first_answer: str = ""
    first_retrieval_top_k: int = 0
    first_context_chars: int = 0
    retry_reason: List[str] = field(default_factory=list)
    retry_top_k: int = 0
    retry_max_chars: int = 0
    query_extra: str = ""
    retry_doc_ids: List[str] = field(default_factory=list)
    retry_context_chars: int = 0
    second_answer: str = ""
    accepted: bool = False
    accept_reason: str = ""
    additional_tokens: int = 0
    additional_prompt_tokens: int = 0
    additional_completion_tokens: int = 0
    key_span_hit_before: float = 0.0
    key_span_hit_after: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return d


def _format_ok(question: str, answer: str, q_type: Optional[str],
               answer_format: Optional[str], option_keys: Sequence[str]) -> Tuple[bool, str]:
    """重答结果的格式验收（只做格式，不判断对错）。"""
    signal = detect_failure(question, answer, q_type=q_type, answer_format=answer_format,
                            option_keys=option_keys, evidence_text="", key_spans=[])
    fmt_reasons = [r for r in signal.reasons if r.startswith("format_violation") or r == "empty_answer"]
    if fmt_reasons:
        return False, ",".join(fmt_reasons)
    return True, "format_ok"


class FailureDrivenQA:
    """首轮作答 → 失败检测 → 定向重检索 → 二次作答，并记录全过程。"""

    def __init__(self, domain: str, base_doc_ids: Optional[Sequence[str]] = None,
                 use_finance_dict: bool = True, page_aware: bool = False,
                 log_path: Optional[str] = None):
        self.domain = domain
        self.base_doc_ids = list(base_doc_ids or [])
        self.use_finance_dict = use_finance_dict
        self.page_aware = page_aware
        self.log_path = log_path or config.RETRY_LOG
        self._pipeline: Optional[DynamicMemoryPipeline] = None

    def pipeline(self) -> DynamicMemoryPipeline:
        if self._pipeline is None:
            self._pipeline = DynamicMemoryPipeline(
                self.domain, self.base_doc_ids,
                use_finance_dict=self.use_finance_dict, page_aware=self.page_aware)
        return self._pipeline

    def run(self, question: str, options: Optional[dict] = None, *, qid: str = "",
            answer_format: Optional[str] = None, q_type: Optional[str] = None,
            budget: Optional[ContextBudget] = None,
            answer_fn: Optional[Callable[[str], Tuple[str, int, int]]] = None,
            candidate_answers: Sequence[str] = ()) -> Tuple[str, RetryRecord, PipelineResult]:
        """返回 (最终答案, 重检索记录, 首轮流水线结果)。

        answer_fn(context) -> (answer, prompt_tokens, completion_tokens)；
        不传 answer_fn 时只做检索侧分析（离线可用，不调用 LLM）。
        """
        options = options or {}
        first = self.pipeline().run(question, options, qid=qid, answer_format=answer_format,
                                    q_type=q_type, budget=budget)
        first_answer = ""
        if answer_fn is not None:
            first_answer, pt, ct = answer_fn(first.context)
            first.prompt_tokens = pt
            first.completion_tokens = ct
            first.token_source = "api"

        signal = detect_failure(question, first_answer, q_type=q_type,
                                answer_format=answer_format,
                                option_keys=list(options.keys()),
                                evidence_text=first.context,
                                key_spans=extract_question_features(
                                    question, options, answer_format=answer_format,
                                    q_type=q_type).key_spans,
                                candidate_answers=candidate_answers)

        record = RetryRecord(
            qid=qid, tier=first.budget.tier, first_answer=first_answer,
            first_retrieval_top_k=first.budget.top_k,
            first_context_chars=first.context_chars,
            retry_reason=list(signal.reasons),
            key_span_hit_before=first.key_span_hit_rate,
        )

        if not signal.is_failure or answer_fn is None:
            if record.retry_reason:
                append_jsonl(self.log_path, record.to_dict())
            return first_answer, record, first

        # ---- 定向重检索 ----
        features = extract_question_features(question, options, qid=qid, domain=self.domain,
                                             doc_ids=self.base_doc_ids,
                                             answer_format=answer_format, q_type=q_type)
        retry_docs = targeted_doc_ids(features, self.domain, self.base_doc_ids)
        retry_top_k = int((budget.top_k if budget else first.budget.top_k) * config.RETRY_TOP_K_MULTIPLIER)
        # 首轮常见失败原因是"预算太小 + 题目里的关键数字根本没被检索到"，
        # 因此重检索同时放大 top_k 与字符上限，并把缺失的关键跨度补进检索词。
        retry_max_chars = int(first.budget.max_chars * 1.5)
        missing_spans = [s.text for s in features.key_spans
                         if s.is_strong and s.text not in first.context][:12]
        query_extra = " ".join(missing_spans)

        retry_pipeline = DynamicMemoryPipeline(self.domain, retry_docs,
                                               use_finance_dict=self.use_finance_dict,
                                               page_aware=self.page_aware)
        retry_budget = ContextBudget(
            tier=first.budget.tier + "+retry", top_k=retry_top_k,
            max_chars=retry_max_chars, compression_ratio=first.budget.compression_ratio,
            use_compression=first.budget.use_compression,
            reasons=list(first.budget.reasons) + [f"retry:{','.join(signal.reasons)}"],
            features=first.budget.features, source="retry")

        second = retry_pipeline.run(question, options, qid=qid, answer_format=answer_format,
                                    q_type=q_type, budget=retry_budget,
                                    query_extra=query_extra)
        second_answer, pt2, ct2 = answer_fn(second.context)
        second.prompt_tokens, second.completion_tokens = pt2, ct2
        second.token_source = "api"

        ok, why = _format_ok(question, second_answer, q_type, answer_format, list(options.keys()))
        record.retry_top_k = retry_top_k
        record.retry_doc_ids = list(retry_docs)
        record.retry_context_chars = second.context_chars
        record.query_extra = query_extra
        record.retry_max_chars = retry_max_chars
        record.second_answer = second_answer
        record.accepted = bool(ok and second_answer)
        record.accept_reason = ("accepted:" + why) if record.accepted else ("rejected:" + why)
        record.additional_prompt_tokens = pt2
        record.additional_completion_tokens = ct2
        record.additional_tokens = pt2 + ct2
        record.key_span_hit_after = second.key_span_hit_rate

        append_jsonl(self.log_path, record.to_dict())
        final = second_answer if record.accepted else first_answer
        return final, record, second


def summarize_retry_log(log_path: Optional[str] = None) -> Dict[str, object]:
    path = log_path or config.RETRY_LOG
    if not os.path.exists(path):
        return {"n_questions": 0, "note": f"日志不存在: {path}"}
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return {"n_questions": 0, "note": "日志为空"}
    n = len(rows)
    reason_counts: Dict[str, int] = {}
    for r in rows:
        for reason in r.get("retry_reason", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "n_questions": n,
        "retry_triggered": sum(1 for r in rows if r.get("retry_reason")),
        "retry_reason_counts": reason_counts,
        "accepted_second_answer": sum(1 for r in rows if r.get("accepted")),
        "additional_tokens": sum(r.get("additional_tokens", 0) for r in rows),
        "avg_additional_tokens": round(sum(r.get("additional_tokens", 0) for r in rows) / n, 1),
    }
