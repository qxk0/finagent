"""文本/题目特征提取（纯规则，确定性，不调用任何 LLM）。

本模块提供两类能力：
1. `extract_spans()`：抽取"关键跨度"——数字、百分比、金额、日期、条款编号、公司/产品实体。
   这些跨度是金融问答里最容易写错、也最容易被压缩弄丢的信息，压缩时必须优先保护。
2. `extract_question_features()`：题面特征，用于动态上下文预算（任务2）。
"""
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


# ---------------------------------------------------------------- 正则定义

_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_MONEY_RE = re.compile(
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*"
    r"(?:万亿元|亿元|亿港元|亿美元|万元|万港元|万美元|元|港元|美元|万)"
)
_DATE_RE = re.compile(
    r"(?:\d{4}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?"
    r"|\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2})"
)
_CLAUSE_RE = re.compile(r"第\s*[一二三四五六七八九十百千零〇两0-9]+\s*(?:条|款|项|章|节|号|部分)")
_TITLE_RE = re.compile(r"《[^》<>]{2,60}》")
_QUOTED_RE = re.compile(r"[“\"']([^”\"']{2,40})[”\"']")
_COMPANY_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9]{2,24}"
    r"(?:股份有限公司|有限责任公司|有限公司|集团|银行|证券|保险|基金|信托|研究院|交易所|事务|事务所)"
)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ASCII_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _normalize(span: str) -> str:
    """跨文档匹配用的归一化：去空白、去千分位逗号、全角转半角。"""
    s = span.strip()
    s = s.replace("，", ",").replace("．", ".").replace("％", "%")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)   # 去千分位
    s = re.sub(r"\s+", "", s)
    # 全角数字 → 半角
    out = []
    for ch in s:
        code = ord(ch)
        if 0xFF10 <= code <= 0xFF19:
            out.append(chr(code - 0xFF10 + ord("0")))
        else:
            out.append(ch)
    return "".join(out)


# 金额/百分比在题目与文档里的写法常不一致（"100万元" vs "100万"，"同比增长12.34%" vs "12.34%"），
# 因此额外生成一个"单位无关"的备用匹配串，降低漏匹配。
_UNIT_SUFFIXES = ("万亿元", "亿元", "万美元", "亿港元", "万元", "万港元", "万亿美元",
                  "元", "美元", "港元", "万")


def _alt_norm(kind: str, norm: str) -> Optional[str]:
    if kind == "money":
        for suf in _UNIT_SUFFIXES:
            if norm.endswith(suf) and len(norm) - len(suf) >= 1:
                return norm[:-len(suf)]
    elif kind == "percent" and norm.endswith("%"):
        return norm[:-1]
    return None


@dataclass(frozen=True)
class Span:
    """一个需要保护的关键信息跨度。"""
    kind: str          # number | percent | money | date | clause | entity | term
    text: str          # 原文形式
    norm: str          # 归一化形式（用于跨文本匹配）
    alt: Optional[str] = None   # 单位无关的备用匹配串，如 "100万元" -> "100万"

    @property
    def is_strong(self) -> bool:
        """强信号：命中即可作为"证据确实相关"的依据。"""
        if self.kind in ("percent", "money", "date", "clause"):
            return True
        if self.kind == "number":
            # 裸数字：>=10 或带小数才视为强信号，避免 "3" 这种到处都有的干扰
            digits = self.norm.replace(".", "")
            return len(self.norm) >= 2 or "." in self.norm or int(digits or 0) >= 10
        return True


def extract_spans(text: str, max_spans: int = 80) -> List[Span]:
    """从任意文本中抽取关键跨度（去重、保序）。"""
    if not text:
        return []

    found: List[Span] = []
    seen = set()

    def _add(kind: str, raw: str):
        norm = _normalize(raw)
        if not norm:
            return
        key = (kind, norm)
        if key in seen:
            return
        seen.add(key)
        found.append(Span(kind=kind, text=raw.strip(), norm=norm, alt=_alt_norm(kind, norm)))

    for m in _PERCENT_RE.finditer(text):
        _add("percent", m.group(0))
    for m in _MONEY_RE.finditer(text):
        _add("money", m.group(0))
    for m in _DATE_RE.finditer(text):
        _add("date", m.group(0))
    for m in _CLAUSE_RE.finditer(text):
        _add("clause", m.group(0))
    for m in _TITLE_RE.finditer(text):
        _add("entity", m.group(0))
    for m in _QUOTED_RE.finditer(text):
        _add("entity", m.group(1))
    for m in _COMPANY_RE.finditer(text):
        _add("entity", m.group(0))

    # 裸数字放最后（优先级低）：只保留前后不是数字/字母/小数点的独立数字
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0)
        start, end = m.span()
        if start > 0 and (text[start - 1].isdigit() or text[start - 1].isalpha()):
            continue
        if end < len(text) and (text[end].isdigit() or text[end].isalpha()):
            continue
        _add("number", raw)

    return found[:max_spans]


def finance_term_hits(text: str, terms: Sequence[str], limit: int = 20) -> List[str]:
    """命中的金融术语（作为额外的关键跨度来源，召回不足时兜底）。"""
    hits = []
    for t in terms:
        if t and t in text:
            hits.append(t)
            if len(hits) >= limit:
                break
    return hits


def span_coverage(spans: Sequence[Span], text: str) -> Dict[str, object]:
    """计算一组关键跨度在 `text` 中的命中情况。"""
    hay = _normalize(text)
    hit, miss = [], []
    for sp in spans:
        matched = sp.norm in hay or (bool(sp.alt) and sp.alt in hay)
        (hit if matched else miss).append(sp)
    total = len(spans)
    return {
        "total": total,
        "hit": hit,
        "miss": miss,
        "hit_rate": (len(hit) / total) if total else 1.0,
        # 强信号命中率：更贴近"证据是否够用"的判断（弱数字不算）
        "strong_total": sum(1 for s in spans if s.is_strong),
        "strong_hit": sum(1 for s in hit if s.is_strong),
    }


def span_in_text(span: Span, text: str) -> bool:
    """单个跨度的命中判断（含单位无关的备用匹配）。"""
    hay = _normalize(text)
    return span.norm in hay or (bool(span.alt) and span.alt in hay)


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    n_cjk = len(_CJK_RE.findall(text))
    return n_cjk / float(len(text))


def char_mix(text: str) -> Dict[str, int]:
    """统计文本的字符构成，用于 token 估计（中文≈1 token/字，ASCII 字母数字更省）。"""
    n_cjk = len(_CJK_RE.findall(text))
    n_ascii_alnum = len(_ASCII_ALNUM_RE.findall(text))
    n_other = max(0, len(text) - n_cjk - n_ascii_alnum)
    return {"cjk": n_cjk, "ascii_alnum": n_ascii_alnum, "other": n_other, "total": len(text)}


# ------------------------------------------------------- 题目特征（动态预算用）

CALC_KEYWORDS = (
    "计算", "求和", "合计", "总额", "差额", "占比", "增幅", "增速", "同比", "环比",
    "多少", "几倍", "增长率", "下降", "上升", "排名", "排序", "从高到低", "从低到高",
    "保留", "小数点", "百分数", "百分号",
)
CROSS_DOC_KEYWORDS = ("分别", "对比", "比较", "各自", "两份", "两份文档", "以下文档", "跨文档", "不同文档", "各公司")

# 分档阈值：取 200 道真题复杂度得分的 p33 / p66
# （实测 simple/normal/complex 三档规模约为 1:1:1，避免某一档只有个位数题目）
SIMPLE_MAX_SCORE = 3.5
NORMAL_MAX_SCORE = 5.0


def score_to_tier(score: float) -> str:
    """复杂度得分 → simple / normal / complex 三档。"""
    if score <= SIMPLE_MAX_SCORE:
        return "simple"
    if score <= NORMAL_MAX_SCORE:
        return "normal"
    return "complex"


@dataclass
class QuestionFeatures:
    qid: str = ""
    domain: str = ""
    n_chars: int = 0
    n_tokens_hint: int = 0
    n_options: int = 0
    n_numbers: int = 0
    n_percents: int = 0
    n_money: int = 0
    n_dates: int = 0
    n_clauses: int = 0
    n_entities: int = 0
    entity_names: List[str] = field(default_factory=list)
    is_choice: bool = False
    is_multi_choice: bool = False
    is_true_false: bool = False
    is_calc: bool = False
    cross_doc: bool = False
    n_docs: int = 0
    complexity_score: float = 0.0
    tier: str = "normal"
    key_spans: List[Span] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "qid": self.qid,
            "domain": self.domain,
            "n_chars": self.n_chars,
            "n_options": self.n_options,
            "n_numbers": self.n_numbers,
            "n_percents": self.n_percents,
            "n_money": self.n_money,
            "n_dates": self.n_dates,
            "n_clauses": self.n_clauses,
            "n_entities": self.n_entities,
            "is_multi_choice": self.is_multi_choice,
            "is_calc": self.is_calc,
            "cross_doc": self.cross_doc,
            "n_docs": self.n_docs,
            "complexity_score": round(self.complexity_score, 2),
            "tier": self.tier,
            "reasons": list(self.reasons),
        }


def extract_question_features(question: str, options: Optional[dict] = None,
                              qid: str = "", domain: str = "",
                              doc_ids: Optional[Sequence[str]] = None,
                              answer_format: Optional[str] = None,
                              q_type: Optional[str] = None) -> QuestionFeatures:
    """抽取题目特征，并给出复杂度打分与 simple/normal/complex 分档。

    打分是规则式的、可解释的（每一条加分都会记录在 reasons 里），
    分档阈值由 `experiments/tune_budget.py` 在真实题目分布上复核。
    """
    options = options or {}
    doc_ids = list(doc_ids or [])
    text = question or ""
    # 选项文本也参与关键跨度抽取：选项里的数字/实体往往就是判定依据
    full_text = text + "\n" + "\n".join(str(v) for v in options.values())

    spans = extract_spans(full_text)
    kinds = [s.kind for s in spans]
    entities = [s.text for s in spans if s.kind == "entity"]

    f = QuestionFeatures(qid=qid, domain=domain, n_chars=len(text))
    f.key_spans = spans
    f.n_numbers = kinds.count("number")
    f.n_percents = kinds.count("percent")
    f.n_money = kinds.count("money")
    f.n_dates = kinds.count("date")
    f.n_clauses = kinds.count("clause")
    f.n_entities = len(entities)
    f.entity_names = entities[:10]
    f.n_options = len(options)
    f.n_docs = len(doc_ids)

    fmt = (answer_format or "").lower()
    tpe = q_type or ""
    f.is_multi_choice = fmt == "multi" or ("多选" in tpe)
    f.is_true_false = fmt == "tf" or ("判断" in tpe)
    f.is_choice = bool(options) and not f.is_true_false
    f.is_calc = (
        tpe in ("计算题", "抽取题")
        or fmt in ("calc", "extract", "numeric", "number")
        or (not options and any(k in text for k in CALC_KEYWORDS))
    )
    f.cross_doc = len(doc_ids) > 1 or any(k in text for k in CROSS_DOC_KEYWORDS)

    # ---- 复杂度打分 ----
    s = 0.0
    if f.is_multi_choice:
        s += 2.0
        f.reasons.append("multi_choice:+2")
    if f.is_calc:
        s += 2.0
        f.reasons.append("requires_calc:+2")
    if f.cross_doc:
        s += 1.5
        f.reasons.append("cross_doc:+1.5")
    if f.n_options >= 4:
        s += 1.0
        f.reasons.append("options>=4:+1")
    if f.n_numbers >= 4:
        s += 1.5
        f.reasons.append("numbers>=4:+1.5")
    elif f.n_numbers >= 2:
        s += 0.5
        f.reasons.append("numbers>=2:+0.5")
    if f.n_percents >= 2:
        s += 1.0
        f.reasons.append("percents>=2:+1")
    if f.n_money >= 2:
        s += 1.0
        f.reasons.append("money>=2:+1")
    if f.n_clauses >= 1:
        s += 0.5
        f.reasons.append("clause_ref:+0.5")
    if f.n_dates >= 1:
        s += 0.5
        f.reasons.append("date_ref:+0.5")
    if f.n_entities >= 3:
        s += 1.0
        f.reasons.append("entities>=3:+1")
    elif f.n_entities >= 1:
        s += 0.5
        f.reasons.append("entities>=1:+0.5")
    if f.n_chars >= 200:
        s += 1.0
        f.reasons.append("long_question:+1")
    elif f.n_chars >= 120:
        s += 0.5
        f.reasons.append("medium_question:+0.5")

    f.complexity_score = s
    f.tier = score_to_tier(s)
    return f


def longest_common_chars(a: str, b: str) -> float:
    """近似重复度：用字符集合 Jaccard 衡量（去重用，避免 O(n^2) 编辑距离）。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sentence_split(text: str, min_len: int = 2) -> List[str]:
    """句子切分：保留标点，绝不切在句子中间。"""
    if not text:
        return []
    parts = re.split(r"(?<=[。；;！!？?])|(?<=})|(?<=\])|\n+", text)
    out = []
    for p in parts:
        if p is None:
            continue
        s = p.strip()
        if len(s) >= min_len:
            out.append(s)
    return out or ([text.strip()] if text.strip() else [])


def idf_weights(docs_tokens: Sequence[Sequence[str]]) -> Dict[str, float]:
    """在候选集内计算 IDF（只用候选块统计，不依赖全库）。"""
    n = len(docs_tokens)
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: math.log(1.0 + n / (1.0 + c)) for t, c in df.items()}
