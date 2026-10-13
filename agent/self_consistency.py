"""多选题自一致性（Multi-choice Self-Consistency）工程化（任务5）。

逻辑与项目原本的多选策略完全一致，不新增任何"复杂 Agent"：
  1. 采样 n 次（默认 3 次，温度 0.1 / 0.3 / 0.3）
  2. 前两次答案一致且非空 → 提前停止（省 token）
  3. 否则第三次参与投票；平票时取字母数较少者（宁缺勿滥，与现有 _vote_multi 一致）

新增的只是**结构化日志**：
  call_count / candidate_answers / early_stop / final_answer / total_tokens
以及聚合统计函数 `summarize_log()`（平均调用次数、提前停止比例、token 成本）。
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import config


def default_vote(candidates: Sequence[str]) -> str:
    """与 reasoner._vote_multi 等价的投票：多数票；平票取字母更少者。"""
    from collections import Counter
    valid = [a for a in candidates if a]
    if not valid:
        return ""
    counter = Counter(valid)
    best = counter.most_common()
    max_count = best[0][1]
    tied = [ans for ans, cnt in best if cnt == max_count]
    if len(tied) == 1:
        return tied[0]
    return sorted(tied, key=lambda x: len(x))[0]


@dataclass
class SelfConsistencyResult:
    final_answer: str
    candidate_answers: List[str] = field(default_factory=list)
    call_count: int = 0
    early_stop: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    temperatures: List[float] = field(default_factory=list)
    responses: List[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, object]:
        return {
            "call_count": self.call_count,
            "candidate_answers": list(self.candidate_answers),
            "distinct_candidates": len({c for c in self.candidate_answers if c}),
            "early_stop": self.early_stop,
            "final_answer": self.final_answer,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "temperatures": list(self.temperatures),
        }


def run_self_consistency(
    call_fn: Callable[[float], Tuple[str, int, int]],
    extract_fn: Callable[[str], str],
    n_samples: int = None,
    temperatures: Sequence[float] = None,
    early_stop_after: int = 2,
    vote_fn: Callable[[Sequence[str]], str] = None,
    qid: str = "",
    log_path: Optional[str] = None,
    meta: Optional[dict] = None,
) -> SelfConsistencyResult:
    """多次采样 + 提前停止 + 投票，并写结构化日志。

    Args:
        call_fn: 接受 temperature，返回 (response_text, prompt_tokens, completion_tokens)
        extract_fn: 从 response_text 抽取答案
        early_stop_after: 前 k 次答案一致即停止（默认 2，与现有实现一致）
    """
    n_samples = n_samples or config.SC_N_SAMPLES
    temperatures = list(temperatures or config.SC_TEMPERATURES)[:n_samples]
    while len(temperatures) < n_samples:
        temperatures.append(temperatures[-1])
    vote_fn = vote_fn or default_vote

    res = SelfConsistencyResult(final_answer="", temperatures=temperatures)

    for i in range(n_samples):
        response, pt, ct = call_fn(temperatures[i])
        res.call_count += 1
        res.prompt_tokens += pt
        res.completion_tokens += ct
        res.responses.append(response or "")
        ans = extract_fn(response) if response else ""
        res.candidate_answers.append(ans)

        if i + 1 >= early_stop_after:
            non_empty = [a for a in res.candidate_answers[-early_stop_after:] if a]
            if len(non_empty) == early_stop_after and len(set(non_empty)) == 1:
                res.early_stop = True
                res.final_answer = non_empty[0]
                break

    if not res.final_answer:
        res.final_answer = vote_fn(res.candidate_answers)

    log_self_consistency(res, qid=qid, log_path=log_path, meta=meta)
    return res


def log_self_consistency(res: SelfConsistencyResult, qid: str = "",
                         log_path: Optional[str] = None,
                         meta: Optional[dict] = None) -> None:
    """追加一行 JSONL 日志：call_count / candidate_answers / early_stop / final_answer / total_tokens。"""
    path = log_path or config.SELF_CONSISTENCY_LOG
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "qid": qid}
        record.update(res.to_dict())
        if meta:
            record.update(meta)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [WARN] 写自一致性日志失败: {e}")


def summarize_log(log_path: Optional[str] = None) -> Dict[str, object]:
    """聚合统计：平均每题调用次数、提前停止比例、token 成本。"""
    path = log_path or config.SELF_CONSISTENCY_LOG
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
    calls = [r.get("call_count", 0) for r in rows]
    early = sum(1 for r in rows if r.get("early_stop"))
    pt = sum(r.get("prompt_tokens", 0) for r in rows)
    ct = sum(r.get("completion_tokens", 0) for r in rows)
    return {
        "n_questions": n,
        "avg_call_count": round(sum(calls) / n, 3),
        "call_count_histogram": {str(k): calls.count(k) for k in sorted(set(calls))},
        "early_stop_ratio": round(early / n, 4),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "avg_total_tokens_per_question": round((pt + ct) / n, 1),
    }
