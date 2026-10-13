"""多选自一致性统计（任务5）。

两种数据来源，优先级从高到低：
  1. `experiments/logs/self_consistency.jsonl`：新版本运行产生的真实日志
     → 直接给出 平均调用次数 / early-stop 比例 / token 成本
  2. 历史运行日志 run_log.txt + 已标定的 token 估计器：
     → 用"该题真实 prompt_tokens ÷ 单次调用估计 token"反推该题调用了几次，
       得到 2 次（提前停止）与 3 次（未提前停止）的比例。
       这是**反推值**，会在输出里明确标注 method=inferred。
"""
import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.self_consistency import summarize_log
from agent.token_estimator import load_estimator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def infer_from_run_log(log_path: str = "run_log.txt",
                       estimator_path: str = None) -> Dict[str, object]:
    """用真实 prompt_tokens / 单次调用估计 token 反推调用次数。"""
    sys.path.insert(0, config.EXPERIMENTS_DIR)
    from calibrate_token_estimator import load_questions, parse_run_log, rebuild_prompt
    from retriever import retrieve_context

    estimator = load_estimator(estimator_path)
    rows = parse_run_log(os.path.join(ROOT, log_path))
    qs = load_questions()
    calls: List[float] = []
    per_q = []
    for r in rows:
        q = qs.get(str(r["qid"]))
        if not q or q.get("answer_format") != "multi":
            continue
        ctx = retrieve_context(q["doc_ids"], q["domain"], q["question"], q["options"],
                               max_chars=12000)
        per_call = estimator.estimate(rebuild_prompt(q, ctx))
        if not per_call:
            continue
        implied = r["prompt_tokens"] / per_call
        calls.append(implied)
        per_q.append({"qid": r["qid"], "observed_prompt_tokens": r["prompt_tokens"],
                      "estimated_single_call_tokens": per_call,
                      "implied_calls": round(implied, 2)})

    n = len(calls)
    if not n:
        return {"method": "inferred", "n_multi_questions": 0}
    two = sum(1 for c in calls if round(c) == 2)
    three = sum(1 for c in calls if round(c) == 3)
    return {
        "method": "inferred",
        "source": log_path,
        "n_multi_questions": n,
        "mean_implied_calls": round(sum(calls) / n, 3),
        "implied_calls_histogram": {"2": two, "3": three,
                                    "other": n - two - three},
        "early_stop_ratio": round(two / n, 4),
        "note": ("反推值：依赖 token 估计器（MAPE≈9%），仅用于在没有新版日志时给出量级；"
                 "真实统计请运行主流程后读取 experiments/logs/self_consistency.jsonl"),
        "per_question": per_q,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=config.SELF_CONSISTENCY_LOG)
    ap.add_argument("--infer", action="store_true", help="没有新版日志时，从历史运行日志反推")
    args = ap.parse_args()

    if os.path.exists(args.log):
        summary = summarize_log(args.log)
        summary["method"] = "real_log"
    elif args.infer:
        summary = infer_from_run_log()
    else:
        summary = {"method": "none",
                   "note": f"未找到 {args.log}；加 --infer 可从历史运行日志反推"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = os.path.join(config.EXPERIMENTS_DIR, "self_consistency_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n结果: {out}")


if __name__ == "__main__":
    main()
