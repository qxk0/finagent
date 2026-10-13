"""自一致性模块的行为与日志测试（不需要 API Key，用桩函数）。

直接运行： python tests/test_self_consistency.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.self_consistency import run_self_consistency, summarize_log


def _stub(answers, tokens=(1000, 20)):
    """按顺序返回预设候选答案的调用桩。"""
    state = {"i": 0}

    def call_fn(temp):
        i = state["i"]
        state["i"] += 1
        ans = answers[min(i, len(answers) - 1)]
        return f"答案：{ans}", tokens[0], tokens[1]
    return call_fn


def test_early_stop():
    log = os.path.join(tempfile.mkdtemp(), "sc.jsonl")
    res = run_self_consistency(_stub(["AC", "AC", "AD"]), extract_fn=_extract,
                               qid="t1", log_path=log)
    assert res.call_count == 2, f"两次一致应提前停止，实际 {res.call_count}"
    assert res.early_stop is True
    assert res.final_answer == "AC"
    assert res.total_tokens == 2 * (1000 + 20)
    with open(log, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    for key in ("call_count", "candidate_answers", "early_stop", "final_answer", "total_tokens"):
        assert key in rec, f"日志缺少字段 {key}"
    assert rec["call_count"] == 2 and rec["early_stop"] is True


def test_no_early_stop_and_vote():
    log = os.path.join(tempfile.mkdtemp(), "sc.jsonl")
    res = run_self_consistency(_stub(["A", "AB", "A"]), extract_fn=_extract,
                               qid="t2", log_path=log)
    assert res.call_count == 3, f"不一致时应调用 3 次，实际 {res.call_count}"
    assert res.early_stop is False
    assert res.final_answer == "A", f"多数票应为 A，实际 {res.final_answer}"
    stats = summarize_log(log)
    assert stats["n_questions"] == 1
    assert stats["avg_call_count"] == 3.0
    assert stats["early_stop_ratio"] == 0.0
    assert stats["total_tokens"] == 3 * 1020


def _extract(resp: str) -> str:
    import re
    m = re.search(r"答案[：:]\s*([A-D]+)", resp)
    return "".join(sorted(set(m.group(1)))) if m else ""


def main():
    tests = [test_early_stop, test_no_early_stop_and_vote]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
