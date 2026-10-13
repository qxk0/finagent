"""失败驱动重检索的离线演示与验证（任务6）。

不需要 API Key：把"首轮作答"用一个**确定性的失败桩**代替（例如返回空答案或缺失百分号的答案），
用来验证失败检测 → 定向重检索 → 二次作答 这条链路真的会触发、真的会换文档、真的会记录日志。
真实答案质量需要 API Key 才能评估（见 --llm 模式下 agent.failure_retrieval.FailureDrivenQA）。

产出：experiments/failure_retrieval_demo.json（逐题：首轮/重检索上下文、命中率变化、重试原因）
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.failure_retrieval import (FailureDrivenQA, detect_failure, targeted_doc_ids)
from agent.text_features import extract_question_features
from evaluate import load_questions

OUT = os.path.join(config.EXPERIMENTS_DIR, "failure_retrieval_demo.json")


def detect_failure_unit_cases() -> List[dict]:
    """失败检测器的单元用例（确定性、可复核）：覆盖每一类可观测信号。"""
    cases = [
        {"case": "空答案", "q": "请计算该公司2024年的营业收入增长率（保留两位小数）。",
         "answer": "", "q_type": "计算题", "opts": {}, "expect": "empty_answer"},
        {"case": "计算题缺 %", "q": "计算该公司2024年的营业收入增长率，结果以百分数表示（保留两位小数）。",
         "answer": "5.55", "q_type": "计算题", "opts": {}, "expect": "format_violation_percent"},
        {"case": "计算题非数字", "q": "计算该公司净利润（保留两位小数）。",
         "answer": "无法计算", "q_type": "计算题", "opts": {}, "expect": "format_violation_numeric"},
        {"case": "小数位不符", "q": "计算资产负债率（保留两位小数）。",
         "answer": "12.3%", "q_type": "计算题", "opts": {}, "expect": "format_violation_decimals"},
        {"case": "选择题答案越界", "q": "下列说法正确的是？",
         "answer": "E", "q_type": "单选题", "opts": {"A": "a", "B": "b"},
         "expect": "format_violation_choice"},
        {"case": "数字证据完全缺失", "q": "若已交保费为987万元、现金价值为654万元，身故保险金是多少？",
         "answer": "987", "q_type": "计算题", "opts": {}, "expect": "numeric_evidence_miss",
         "evidence": "第三条 被保险人身故的，本公司按基本保险金额的120%给付身故保险金。"},
        {"case": "多次采样不一致", "q": "下列说法正确的有？", "answer": "AC",
         "q_type": "多选题", "opts": {"A": "a", "B": "b", "C": "c", "D": "d"},
         "candidates": ["AC", "AD", "AC"], "expect": "self_consistency_disagreement"},
        {"case": "正常答案（不应报错）", "q": "计算资产负债率（保留两位小数）。",
         "answer": "12.34%", "q_type": "计算题", "opts": {},
         "evidence": "资产负债率为12.34%，较上年下降1.02个百分点。", "expect": "NONE"},
    ]
    out = []
    for c in cases:
        key_spans = extract_question_features(c["q"], c["opts"], q_type=c["q_type"]).key_spans
        sig = detect_failure(c["q"], c["answer"], q_type=c["q_type"],
                             option_keys=list(c["opts"].keys()),
                             evidence_text=c.get("evidence", c["q"]),
                             key_spans=key_spans,
                             candidate_answers=c.get("candidates", ()))
        reasons = sig.reasons or ["NONE"]
        out.append({"case": c["case"], "expected": c["expect"], "detected": reasons,
                    "passed": c["expect"] in reasons})
        print(f"  [{'OK' if c['expect'] in reasons else 'XX'}] {c['case']}: 期望={c['expect']} 实际={reasons}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="B", choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--only-calc", action="store_true", default=True)
    ap.add_argument("--offline", action="store_true", default=True,
                    help="用确定性失败桩代替真实模型输出（无需 API Key）")
    args = ap.parse_args()

    questions = load_questions(args.split)
    if args.only_calc:
        questions = [q for q in questions if (q.get("type") in ("计算题", "抽取题")
                                              or not q.get("options"))]
    questions = questions[:args.limit]
    print(f"选取 {len(questions)} 道计算/抽取题进行失败驱动重检索演示")

    print("\n--- 失败检测器单元用例 ---")
    unit_cases = detect_failure_unit_cases()

    print("\n--- 定向重检索链路演示 ---")
    results = []
    t0 = time.time()
    for q in questions:
        qid = q["qid"]
        feats = extract_question_features(q["question"], q.get("options") or {}, qid=qid,
                                          domain=q["domain"], doc_ids=q.get("doc_ids") or [],
                                          answer_format=q.get("answer_format"), q_type=q.get("type"))
        # 失败桩：首轮返回空字符串 → 必然触发 empty_answer；重试轮返回 "0" → 通过格式验收
        calls = {"n": 0}

        def stub_answer(context):
            calls["n"] += 1
            if calls["n"] == 1:
                return "", 0, 0
            return "0", 0, 0

        qa = FailureDrivenQA(q["domain"], q.get("doc_ids") or [])
        final, record, second = qa.run(q["question"], q.get("options") or {}, qid=qid,
                                       answer_format=q.get("answer_format"),
                                       q_type=q.get("type"), answer_fn=stub_answer)
        info = {
            "qid": qid, "domain": q["domain"], "q_type": q.get("type"),
            "retry_reason": record.retry_reason,
            "first_retrieval_top_k": record.first_retrieval_top_k,
            "retry_top_k": record.retry_top_k,
            "first_context_chars": record.first_context_chars,
            "retry_context_chars": record.retry_context_chars,
            "first_answer": record.first_answer, "second_answer": record.second_answer,
            "accepted": record.accepted, "accept_reason": record.accept_reason,
            "additional_tokens": record.additional_tokens,
            "retry_doc_ids": record.retry_doc_ids[:6],
            "entities_used_for_targeting": feats.entity_names[:6],
            "key_span_hit_before": record.key_span_hit_before,
            "key_span_hit_after": record.key_span_hit_after,
            "final_answer_returned": final,
        }
        results.append(info)
        print(f"  [{qid}] reasons={record.retry_reason} top_k {record.first_retrieval_top_k}"
              f"→{record.retry_top_k} chars {record.first_context_chars}→{record.retry_context_chars}"
              f" 命中率 {record.key_span_hit_before:.2f}→{record.key_span_hit_after:.2f}"
              f" docs={record.retry_doc_ids[:3]}")

    summary = {
        "split": args.split,
        "n_questions": len(results),
        "n_retry_triggered": sum(1 for r in results if r["retry_reason"]),
        "reasons_histogram": {},
        "avg_retry_top_k": (sum(r["retry_top_k"] for r in results) / len(results)) if results else 0,
        "avg_context_chars_gain": (sum(r["retry_context_chars"] - r["first_context_chars"] for r in results)
                                   / len(results)) if results else 0,
        "avg_key_span_hit_gain": (sum(r["key_span_hit_after"] - r["key_span_hit_before"] for r in results)
                                  / len(results)) if results else 0,
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": ("离线桩演示：只验证失败检测与定向重检索链路会触发并生效；"
                 "真实答案正确率需要 API Key 运行 --llm 场景"),
    }
    hist: Dict[str, int] = {}
    for r in results:
        for reason in r["retry_reason"]:
            hist[reason] = hist.get(reason, 0) + 1
    summary["reasons_histogram"] = hist

    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detector_unit_cases": unit_cases,
                   "per_question": results}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果: {OUT}")


if __name__ == "__main__":
    main()
