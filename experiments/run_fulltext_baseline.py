"""Full-text baseline（任务4）：Question + 相关完整文档 → Qwen → Answer。

严格按题意实现：**不做 BM25 检索、不做记忆压缩**，把题目关联文档的全文直接放进 prompt。

现实约束（实测，不是推测）：
  * A 榜单题关联文档平均 386,249 字符（最大 1,463,900），远超 qwen-plus 的
    128K 上下文窗口 → 这一类题目的全文基线在物理上无法执行，脚本会如实记录
    `SKIPPED_CONTEXT_LIMIT` 并保留实测字符数，绝不通过截断"假装"跑通。
  * 只有文档量能塞进上下文窗口的题目才真正调用模型，并统计真实 token。

产物：experiments/baseline_fulltext.csv（逐题）、experiments/baseline_fulltext_summary.json
"""
import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.corpus import domain_doc_ids
from agent.pipeline import append_jsonl, build_fulltext_context
from agent.token_estimator import load_estimator
from evaluate import load_labels, load_questions, normalize_answer, question_kind

def domain_corpus_chars(domain: str) -> int:
    """领域全量语料字符数（B 榜没有 doc_ids 时用于判断基线是否可行）。"""
    total = 0
    for doc_id in domain_doc_ids(domain):
        from preprocess import load_document
        total += len(load_document(doc_id, domain) or "")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A", choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", action="store_true", help="真实调用 Qwen（否则只测量上下文规模）")
    args = ap.parse_args()

    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.EXPERIMENTS_DIR, f"baseline_fulltext_{args.split.lower()}.csv")
    summary_path = os.path.join(config.EXPERIMENTS_DIR,
                                f"baseline_fulltext_summary_{args.split.lower()}.json")
    questions = load_questions(args.split)[:args.limit] if args.limit else load_questions(args.split)
    labels = load_labels()
    estimator = load_estimator()
    print(f"{args.split} 榜 {len(questions)} 题；模式={'LLM' if args.llm else '仅测量'}")

    from main_b import (SYSTEM_PROMPT_CHOICE, SYSTEM_PROMPT_CALC, build_choice_prompt,
                        build_calc_prompt, MAX_TOKENS_CHOICE, MAX_TOKENS_CALC)

    rows: List[Dict[str, object]] = []
    n_feasible = n_skipped = 0
    total_pt = total_ct = 0
    scored = []
    hits = 0

    domain_chars_cache: Dict[str, int] = {}
    for i, q in enumerate(questions, 1):
        qid, domain = q["qid"], q["domain"]
        kind = question_kind(q)
        options = q.get("options") or {}
        row: Dict[str, object] = {"qid": qid, "domain": domain, "kind": kind}

        doc_ids = q.get("doc_ids") or []
        if doc_ids:
            ctx, meta = build_fulltext_context(doc_ids, domain)
            row["n_docs"] = meta["n_docs"]
        else:
            # B 榜无 doc_ids：严格意义的"相关完整文档"不可知，用领域全量语料规模衡量
            if domain not in domain_chars_cache:
                domain_chars_cache[domain] = domain_corpus_chars(domain)
            row["n_docs"] = len(domain_doc_ids(domain))
            row["note"] = "B榜无 doc_ids，按领域全量语料规模估算"
            ctx = ""
            meta = {"n_docs": row["n_docs"], "chars": domain_chars_cache[domain], "context_chars": 0}

        row["doc_chars"] = meta["chars"]
        if not ctx and meta["chars"]:
            # 用实际字符数做规模判断（不真正拼接超大文本，避免内存爆炸）
            est_pt = estimator.estimate("中" * meta["chars"])
        else:
            task_prompt = (build_choice_prompt(q["question"], options, q.get("type") or "单选题", ctx)
                           if kind in ("single", "multi", "tf")
                           else build_calc_prompt(q["question"], ctx))
            est_pt = estimator.estimate_prompt(SYSTEM_PROMPT_CHOICE, task_prompt)
        row["estimated_prompt_tokens"] = est_pt
        row["context_chars"] = len(ctx)

        feasible = est_pt <= config.FULLTEXT_MAX_INPUT_TOKENS and bool(ctx)
        if not feasible:
            row["status"] = "SKIPPED_CONTEXT_LIMIT"
            n_skipped += 1
            rows.append(row)
            if i % 20 == 0:
                print(f"  {i}/{len(questions)} 完成", flush=True)
            continue

        row["status"] = "RUN"
        n_feasible += 1
        if args.llm:
            from experiments.run_ablation import answer_with_qwen
            system_prompt = SYSTEM_PROMPT_CHOICE if kind in ("single", "multi", "tf") else SYSTEM_PROMPT_CALC
            max_tokens = MAX_TOKENS_CHOICE if kind in ("single", "multi", "tf") else MAX_TOKENS_CALC
            ans, pt, ct, _ = answer_with_qwen(ctx, q, system_prompt, task_prompt, max_tokens, False, qid)
            row.update({"answer": ans, "prompt_tokens": pt, "completion_tokens": ct,
                        "total_tokens": pt + ct, "token_source": "api"})
            total_pt += pt
            total_ct += ct
            truth = labels.get(qid)
            if truth:
                row["gold"] = truth
                hit = int(normalize_answer(ans, q.get("answer_format"))
                          == normalize_answer(truth, q.get("answer_format")))
                row["hit"] = hit
                hits += hit
                scored.append(hit)
        else:
            row.update({"answer": "", "prompt_tokens": est_pt, "completion_tokens": 0,
                        "total_tokens": est_pt, "token_source": "estimated"})
            total_pt += est_pt
        rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(questions)} 完成", flush=True)

    fields = ["qid", "domain", "kind", "status", "n_docs", "doc_chars", "context_chars",
              "estimated_prompt_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
              "token_source", "answer", "gold", "hit", "note"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "split": args.split,
        "n_questions": len(rows),
        "n_feasible": n_feasible,
        "n_skipped": n_skipped,
        "skip_reason": "全文超过模型上下文窗口（128K tokens）",
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
        "token_source": "api" if args.llm else "estimated",
        "accuracy_on_feasible": (round(hits / len(scored), 4) if scored else None),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n逐题: {csv_path}\n汇总: {summary_path}")


if __name__ == "__main__":
    main()
