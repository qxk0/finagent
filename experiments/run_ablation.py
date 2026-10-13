"""消融实验（任务4 + 任务7）。

对比组：
  A. full_text            严格基线：相关文档全文直接入 prompt（不用 BM25、不压缩）
  B. bm25                 纯 jieba BM25（沿用领域固定 top_k / max_chars）
  C. bm25_finance         BM25 + 133 术语金融词典（= 项目原有系统）
  D. bm25_dynamic         C + 动态上下文预算（simple/normal/complex）
  E. dynamic_compress     D + 动态记忆压缩 ← 本次新增的核心能力
  F. mc_dynamic_compress  E + 多选题自一致性多次采样

两种运行模式：
  --no-llm   只跑检索/压缩链路：真实测量 context_chars、selected_chunks、
             compression_ratio、关键跨度覆盖率，token 用已标定的估计器给出（标记 estimated）
  --llm      真实调用 Qwen，得到真实 prompt/completion tokens 与答案

产物：
  experiments/results.csv                 每个实验一行的汇总（README 表格的数据源）
  experiments/details/<split>_<config>.jsonl  每题明细（top_k/max_chars/selected_chunks/context_chars/…）
                                              split=a|b，A/B 榜分开命名，避免互相覆盖
  experiments/ablation_summary.json       含 token_saving_rate 与说明
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.context_budget import ContextBudget, load_tier_params
from agent.pipeline import (DynamicMemoryPipeline, append_jsonl, build_fulltext_context)
from agent.self_consistency import run_self_consistency
from agent.token_estimator import load_estimator
from retriever import BM25Retriever, build_query
from evaluate import load_labels, load_questions, normalize_answer, question_kind

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DETAIL_DIR = os.path.join(config.EXPERIMENTS_DIR, "details")


def detail_path_for(split: str, cfg_name: str) -> str:
    """逐题明细路径：带榜单前缀，避免 A/B 榜同名互相覆盖。"""
    return os.path.join(DETAIL_DIR, f"{split.lower()}_{cfg_name}.jsonl")

CONFIGS = {
    "A_full_text": {"desc": "Full-text baseline（相关文档全文直入）", "kind": "full_text"},
    "B_bm25": {"desc": "BM25（纯 jieba，领域固定预算）", "kind": "legacy", "finance_dict": False},
    "C_bm25_finance": {"desc": "BM25 + 金融词典（领域固定预算）", "kind": "legacy", "finance_dict": True},
    "D_bm25_dynamic": {"desc": "BM25 + 金融词典 + 动态上下文预算", "kind": "dynamic",
                       "finance_dict": True, "compress": False},
    "E_dynamic_compress": {"desc": "D + 动态记忆压缩", "kind": "dynamic",
                           "finance_dict": True, "compress": True},
    "F_mc_dynamic_compress": {"desc": "E + 多选自一致性", "kind": "dynamic",
                              "finance_dict": True, "compress": True, "self_consistency": True},
}


def load_data(split: str) -> List[dict]:
    return load_questions(split)


# ---------------------------------------------------------------- 上下文构建

def legacy_context(q: dict, finance_dict: bool) -> Tuple[str, Dict[str, object]]:
    """原有系统：领域固定 top_k / max_chars 的 BM25 上下文。"""
    doc_ids = q.get("doc_ids") or []
    domain = q["domain"]
    if not doc_ids:
        from main_b import DomainRetriever
        r = DomainRetriever.get(domain, use_finance_dict=finance_dict)
        query = build_query(q["question"], q.get("options") or {}) if q.get("options") else q["question"]
        ctx = r.retrieve(query)
        return ctx, {"top_k": config.DOMAIN_CONFIG.get(domain, {}).get("top_k", config.TOP_K),
                     "max_chars": config.DOMAIN_CONFIG.get(domain, {}).get("max_ctx", config.MAX_CONTEXT_CHARS),
                     "n_selected": None, "chars_before": len(ctx)}
    retr = BM25Retriever(doc_ids, domain, use_finance_dict=finance_dict)
    query = build_query(q["question"], q.get("options") or {}) if q.get("options") else q["question"]
    cfg = config.DOMAIN_CONFIG.get(domain, {})
    top_k = cfg.get("top_k", config.TOP_K)
    max_chars = cfg.get("max_ctx", config.MAX_CONTEXT_CHARS)
    chunks = retr.retrieve(query, top_k=top_k, max_chars=max_chars)
    ctx = "\n\n".join(f"[证据{i+1}]\n{c}" for i, c in enumerate(chunks))
    return ctx, {"top_k": top_k, "max_chars": max_chars,
                 "n_selected": len(chunks), "chars_before": sum(len(c) for c in chunks)}


def answer_with_qwen(context: str, q: dict, system_prompt: str, user_prompt: str,
                     max_tokens: int, self_consistency: bool, qid: str):
    """统一作答入口；返回 (answer, prompt_tokens, completion_tokens, sc_stats)。"""
    from main_b import (extract_choice_answer, extract_calc_answer, call_llm_b,
                        split_multi_answers)

    kind = question_kind(q)
    if kind in ("single", "multi", "tf"):
        if kind == "multi" and self_consistency:
            sc = run_self_consistency(
                call_fn=lambda temp: call_llm_b(system_prompt, user_prompt,
                                                max_tokens=max_tokens, temperature=temp),
                extract_fn=lambda resp: extract_choice_answer(resp, q.get("type") or "多选题"),
                qid=qid, log_path=config.SELF_CONSISTENCY_LOG,
                meta={"pipeline": "ablation"})
            return sc.final_answer, sc.prompt_tokens, sc.completion_tokens, sc.to_dict()
        resp, pt, ct = call_llm_b(system_prompt, user_prompt, max_tokens=max_tokens)
        ans = extract_choice_answer(resp, q.get("type") or "单选题")
        return ans, pt, ct, {"call_count": 1, "early_stop": False,
                             "candidate_answers": [ans], "final_answer": ans,
                             "total_tokens": pt + ct}
    resp, pt, ct = call_llm_b(system_prompt, user_prompt, max_tokens=max_tokens)
    ans = extract_calc_answer(resp, q["question"])
    return ans, pt, ct, {"call_count": 1, "early_stop": False,
                         "candidate_answers": [ans], "final_answer": ans,
                         "total_tokens": pt + ct}


# ---------------------------------------------------------------- 单组实验

def run_config(cfg_name: str, cfg: dict, questions: List[dict], labels: Dict[str, str],
               use_llm: bool, limit: Optional[int] = None,
               tier_params: Optional[dict] = None,
               split: str = "A") -> Dict[str, object]:
    from main_b import (SYSTEM_PROMPT_CHOICE, SYSTEM_PROMPT_CALC, build_choice_prompt,
                        build_calc_prompt, MAX_TOKENS_CHOICE, MAX_TOKENS_CALC)

    # 明细文件名必须带榜单前缀：A/B 榜的题号不同，共用文件名会互相覆盖
    # （曾导致 A 榜的 B/C/D/E 明细被 B 榜同名文件覆盖，results.csv 无法回溯）
    detail_path = detail_path_for(split, cfg_name)
    if os.path.exists(detail_path):
        os.remove(detail_path)          # 每次实验重跑，避免新旧结果混杂
    estimator = load_estimator()
    qs = questions[:limit] if limit else questions
    rows = []
    t0 = time.time()

    for i, q in enumerate(qs, 1):
        qid = q["qid"]
        kind = question_kind(q)
        options = q.get("options") or {}
        row: Dict[str, object] = {"qid": qid, "domain": q["domain"], "kind": kind,
                                  "config": cfg_name, "skipped": ""}

        # ---------- 构造上下文 ----------
        if cfg["kind"] == "full_text":
            ctx, meta = build_fulltext_context(q.get("doc_ids") or [], q["domain"])
            row.update(meta)
            # 先估算 prompt token，判断是否超出模型上下文窗口
            task_prompt = (build_choice_prompt(q["question"], options, q.get("type") or "单选题", ctx)
                           if kind in ("single", "multi", "tf")
                           else build_calc_prompt(q["question"], ctx))
            est_pt = estimator.estimate_prompt(SYSTEM_PROMPT_CHOICE, task_prompt)
            row["context_chars"] = len(ctx)
            row["estimated_prompt_tokens"] = est_pt
            if est_pt > config.FULLTEXT_MAX_INPUT_TOKENS:
                row["skipped"] = "SKIPPED_CONTEXT_LIMIT"
                rows.append(row)
                append_jsonl(detail_path, row)
                continue
        elif cfg["kind"] == "legacy":
            ctx, meta = legacy_context(q, cfg.get("finance_dict", True))
            row.update({k: v for k, v in meta.items()})
            row["context_chars"] = len(ctx)
            row["compression_ratio"] = 1.0
            task_prompt = (build_choice_prompt(q["question"], options, q.get("type") or "单选题", ctx)
                           if kind in ("single", "multi", "tf")
                           else build_calc_prompt(q["question"], ctx))
            row["estimated_prompt_tokens"] = estimator.estimate_prompt(SYSTEM_PROMPT_CHOICE, task_prompt)
        else:
            if q.get("doc_ids"):
                pipe = DynamicMemoryPipeline(q["domain"], q["doc_ids"],
                                             use_finance_dict=cfg.get("finance_dict", True),
                                             tier_params=tier_params)
            else:
                pipe = DynamicMemoryPipeline(q["domain"], None,
                                             use_finance_dict=cfg.get("finance_dict", True),
                                             share_domain_index=True,
                                             tier_params=tier_params)
            res = pipe.run(q["question"], options, qid=qid,
                           answer_format=q.get("answer_format"), q_type=q.get("type"),
                           use_compression=cfg.get("compress", True),
                           system_prompt=SYSTEM_PROMPT_CHOICE)
            ctx = res.context
            row.update({
                "tier": res.budget.tier, "top_k": res.budget.top_k,
                "max_chars": res.budget.max_chars,
                "n_candidates": res.n_candidates, "n_selected": len(res.selected),
                "selected_chunks": [s.to_dict() for s in res.selected],
                "context_chars": res.context_chars,
                "chars_before_compression": res.chars_before_compression,
                "compression_ratio": round(res.compression_ratio, 4),
                "key_span_hit_rate": round(res.key_span_hit_rate, 4),
                "compression_stats": res.compression_stats,
                "traceability_problems": res.traceability_problems,
            })
            task_prompt = (build_choice_prompt(q["question"], options, q.get("type") or "单选题", ctx)
                           if kind in ("single", "multi", "tf")
                           else build_calc_prompt(q["question"], ctx))
            row["estimated_prompt_tokens"] = estimator.estimate_prompt(SYSTEM_PROMPT_CHOICE, task_prompt)

        # ---------- 作答 ----------
        if use_llm:
            system_prompt = (SYSTEM_PROMPT_CHOICE if kind in ("single", "multi", "tf")
                             else SYSTEM_PROMPT_CALC)
            max_tokens = MAX_TOKENS_CHOICE if kind in ("single", "multi", "tf") else MAX_TOKENS_CALC
            ans, pt, ct, sc = answer_with_qwen(ctx, q, system_prompt, task_prompt,
                                               max_tokens, cfg.get("self_consistency", False), qid)
            row.update({"answer": ans, "prompt_tokens": pt, "completion_tokens": ct,
                        "total_tokens": pt + ct, "token_source": "api",
                        "self_consistency": sc})
        else:
            row.update({"answer": "", "prompt_tokens": row["estimated_prompt_tokens"],
                        "completion_tokens": 0,
                        "total_tokens": row["estimated_prompt_tokens"],
                        "token_source": "estimated"})

        truth = labels.get(qid)
        if truth:
            row["gold"] = truth
            row["hit"] = int(normalize_answer(row.get("answer", ""), q.get("answer_format"))
                             == normalize_answer(truth, q.get("answer_format")))
        rows.append(row)
        append_jsonl(detail_path, row)
        if i % 20 == 0:
            print(f"  [{cfg_name}] {i}/{len(qs)}  ({time.time() - t0:.0f}s)", flush=True)

    # ---------- 汇总 ----------
    done = [r for r in rows if not r.get("skipped")]
    scored = [r for r in done if "hit" in r]
    summary = {
        "config": cfg_name,
        "description": cfg["desc"],
        "n_questions": len(rows),
        "n_skipped": len(rows) - len(done),
        "skip_reason": "SKIPPED_CONTEXT_LIMIT（全文超出模型上下文窗口）" if len(rows) != len(done) else "",
        "n_scored": len(scored),
        "accuracy": (round(sum(r["hit"] for r in scored) / len(scored), 4) if scored else None),
        "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in done),
        "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in done),
        "avg_context_chars": (round(sum(int(r.get("context_chars") or 0) for r in done) / len(done), 1)
                              if done else 0),
        "avg_selected_chunks": (round(sum(int(r.get("n_selected") or 0) for r in done) / len(done), 2)
                                if done else 0),
        "avg_compression_ratio": (round(sum(float(r.get("compression_ratio") or 1.0) for r in done)
                                        / len(done), 4) if done else 1.0),
        "token_source": ("api" if use_llm else "estimated"),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    summary["total_tokens"] = summary["prompt_tokens"] + summary["completion_tokens"]
    if cfg.get("self_consistency"):
        calls = [r["self_consistency"].get("call_count", 1) for r in done if r.get("self_consistency")]
        early = [1 for r in done if r.get("self_consistency", {}).get("early_stop")]
        summary["avg_call_count"] = round(sum(calls) / len(calls), 3) if calls else None
        summary["early_stop_ratio"] = round(sum(early) / len(calls), 4) if calls else None
    print(f"  [{cfg_name}] 完成: {summary['n_questions']} 题, "
          f"avg_context={summary['avg_context_chars']} 字符, tokens={summary['total_tokens']:,}",
          flush=True)
    return summary


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A", choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--configs", default=",".join(CONFIGS.keys()))
    ap.add_argument("--llm", action="store_true", help="真实调用 Qwen（否则只跑检索/压缩链路）")
    ap.add_argument("--out", default=config.RESULTS_CSV)
    args = ap.parse_args()

    os.makedirs(DETAIL_DIR, exist_ok=True)
    questions = load_data(args.split)
    labels = load_labels()
    print(f"{args.split} 榜题目 {len(questions)} 道；gold label {len(labels)} 条；"
          f"模式={'LLM' if args.llm else '离线'}")

    chosen = [c.strip() for c in args.configs.split(",") if c.strip()]
    tiers_path = (config.BUDGET_TIERS_FILE_B if args.split == "B" else config.BUDGET_TIERS_FILE)
    tier_params = load_tier_params(tiers_path)
    print(f"预算参数来源: {tiers_path}")
    summaries = []
    for name in chosen:
        if name not in CONFIGS:
            print(f"[WARN] 未知实验组: {name}")
            continue
        print(f"\n=== {name}: {CONFIGS[name]['desc']} ===", flush=True)
        summaries.append(run_config(name, CONFIGS[name], questions, labels,
                                    use_llm=args.llm, limit=args.limit,
                                    tier_params=tier_params, split=args.split))

    # ---- token 节省率：必须**在同一个题目子集**上比较 ----
    # 全文基线有 48/100 题放不进模型上下文窗口，拿它的总量去和全量系统比是错的。
    # 因此这里取"全文基线可执行"的题目集合，各实验组只在该子集上汇总 token。
    feasible_qids = set()
    current_qids = {q["qid"] for q in questions}
    a_detail = detail_path_for(args.split, "A_full_text")
    if os.path.exists(a_detail):
        for line in open(a_detail, "r", encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if not r.get("skipped") and r["qid"] in current_qids:
                    feasible_qids.add(r["qid"])
    if not feasible_qids:
        # 没有跑全文基线时，用专门的基线脚本文档（experiments/baseline_fulltext_<split>.csv）
        bl = os.path.join(config.EXPERIMENTS_DIR, f"baseline_fulltext_{args.split.lower()}.csv")
        if os.path.exists(bl):
            for r in csv.DictReader(open(bl, encoding="utf-8")):
                if r.get("status") == "RUN" and r["qid"] in current_qids:
                    feasible_qids.add(r["qid"])

    def subset_tokens(cfg_name: str) -> int:
        path = detail_path_for(args.split, cfg_name)
        if not os.path.exists(path):
            return 0
        total = 0
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("skipped") or r["qid"] not in feasible_qids:
                continue
            total += int(r.get("total_tokens") or 0)
        return total

    baseline_subset = subset_tokens("A_full_text")
    for s in summaries:
        sub = subset_tokens(s["config"])
        s["n_fulltext_subset"] = len(feasible_qids)
        s["tokens_on_fulltext_subset"] = sub
        s["token_saving_rate_vs_fulltext"] = (
            round(1 - sub / baseline_subset, 4) if baseline_subset and sub else None)

    fields = ["config", "description", "n_questions", "n_skipped", "n_scored", "accuracy",
              "prompt_tokens", "completion_tokens", "total_tokens", "token_source",
              "avg_context_chars", "avg_selected_chunks", "avg_compression_ratio",
              "avg_call_count", "early_stop_ratio", "token_saving_rate_vs_fulltext",
              "n_fulltext_subset", "tokens_on_fulltext_subset",
              "skip_reason", "elapsed_seconds"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\n结果表: {args.out}")

    with open(os.path.join(config.EXPERIMENTS_DIR, "ablation_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "split": args.split, "llm": args.llm,
                   "limit": args.limit, "summaries": summaries},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
