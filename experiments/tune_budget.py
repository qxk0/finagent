"""动态预算 / 压缩参数选择（离线，不调用任何 LLM）。

为什么用"关键跨度覆盖率"作为目标：
  当前数据集**没有任何标准答案**（见 experiments/labels_audit.json），
  无法直接用 accuracy 选参数。这里用一个与正确性正相关的无标签代理指标：
  题目里的关键跨度（金额/百分比/日期/条款号/多位数字/实体）在检索上下文中的覆盖率。
  同时分别统计"仅题干跨度"和"题干+选项跨度"两个版本——选项里的数字常是干扰项，
  题干版本更贴近"作答真正需要的信息"。

选择规则（可复现的肘部/饱和规则，而非拍脑袋）：
  1) 固定字符上限为网格最大值，扫 top_k：取覆盖率**饱和**的最小 k*
     （其后两个网格点的提升均 ≤ 1%）。
  2) 固定 k*，扫字符上限：取覆盖率相对无上限下降 ≤ 1% 的最小 c*。
  3) 压缩比：在选定的 (k*, c*) 下扫 compression_ratio，
     取"关键跨度保护率 = 100% 且回退块占比 ≤ 50%"的最小压缩比。

产出：
  experiments/budget_tuning_{split}.csv    (top_k × max_chars 覆盖率网格)
  experiments/compression_tuning_{split}.csv
  experiments/budget_tiers.json            (A 榜最终三档参数，pipeline 直接读取)
  experiments/budget_tuning_b.json         (B 榜参数，供 B 榜流水线复用)
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.corpus import domain_doc_ids
from agent.memory_compressor import compress_evidence, filter_evidence, score_evidence_chunks
from agent.pipeline import DynamicMemoryPipeline
from agent.text_features import extract_question_features, idf_weights, span_coverage
from agent.token_estimator import load_estimator
from retriever import FINANCE_TERMS, build_query, tokenize

TOP_K_GRID = [5, 8, 10, 12, 15, 20, 25, 30, 40, 60]
CHAR_GRID = [4000, 6000, 8000, 10000, 12000, 16000, 20000, 28000, 40000]
RATIO_GRID = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]
COVERAGE_FLOOR = 0.95          # 取到参考覆盖率(60块/无上限)的 95%
REFERENCE_TOP_K = 60


def load_questions(split: str = "A") -> List[dict]:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if split == "A":
        out = []
        for p in sorted(glob.glob(os.path.join(config.QUESTIONS_DIR, "*.json"))):
            out += json.load(open(p, encoding="utf-8"))
        return out
    out = []
    b_dir = os.path.join(root, "upload_b", "question_b")
    for p in sorted(glob.glob(os.path.join(b_dir, "*"))):
        if p.endswith(".jsonl"):
            out += [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
        elif p.endswith(".json"):
            out += json.load(open(p, encoding="utf-8-sig"))
    return out


def rank_chunks(pipe: DynamicMemoryPipeline, question: str, options: dict, n_candidates: int = 200):
    query = build_query(question, options) if options else question
    cands = pipe.candidate_chunks(query, n_candidates=n_candidates)
    if not cands:
        return [], {}, query
    toks = [tokenize(c.text, use_finance_dict=pipe.use_finance_dict) for c in cands]
    idf = idf_weights(toks)
    ranked = score_evidence_chunks(cands, [], tokenize(query, use_finance_dict=pipe.use_finance_dict),
                                   idf, finance_terms=FINANCE_TERMS)
    ranked = filter_evidence(ranked, top_k=REFERENCE_TOP_K * 2)
    return ranked, idf, query


def coverage_for(ranked, key_spans, top_k: int, max_chars: Optional[int]) -> float:
    parts, used = [], 0
    for ev in ranked[:top_k]:
        if max_chars is not None and used + len(ev.text) > max_chars and parts:
            continue
        parts.append(ev.text)
        used += len(ev.text)
    if not parts or not key_spans:
        return 0.0
    return span_coverage(key_spans, "\n".join(parts))["hit_rate"]


def choose_anchor(pooled_rel: Dict[tuple, float], floor: float = COVERAGE_FLOOR) -> tuple:
    """在**全题池**的覆盖率曲线上取膝点：成本优先，第一个达到 floor 的 (max_chars, top_k)。"""
    for c in sorted(CHAR_GRID):
        for k in sorted(TOP_K_GRID):
            if pooled_rel.get((k, c), 0.0) >= floor:
                return k, c, pooled_rel[(k, c)]
    kc = max(pooled_rel, key=lambda x: pooled_rel[x])
    return kc[0], kc[1], pooled_rel[kc]


def snap_to_grid(value: float, grid) -> int:
    return min(grid, key=lambda g: abs(g - value))


def build_tier_ladder(anchor_top_k: int, anchor_chars: int) -> Dict[str, Dict[str, int]]:
    """以膝点为锚构造 simple/normal/complex 阶梯（0.5x / 1x / 1.5x）。

    为什么不在每档内部独立选参数：分档后的样本量（十几到几十题）会让
    "关键跨度覆盖率"出现量化跳变（某一题从 0.5 跳到 1.0），选出的参数不稳定。
    因此以全题池膝点为锚，三档按固定倍数缩放，并把每档实测覆盖率如实报告。
    """
    return {
        "simple": {"top_k": snap_to_grid(anchor_top_k / 2, TOP_K_GRID),
                   "max_chars": snap_to_grid(anchor_chars / 2, CHAR_GRID)},
        "normal": {"top_k": anchor_top_k, "max_chars": anchor_chars},
        "complex": {"top_k": snap_to_grid(anchor_top_k * 1.5, TOP_K_GRID),
                    "max_chars": snap_to_grid(anchor_chars * 1.5, CHAR_GRID)},
    }


def tune(split: str, limit: Optional[int] = None) -> Dict[str, object]:
    questions = load_questions(split)
    if limit:
        questions = questions[:limit]
    print(f"题目数: {len(questions)} (split={split})", flush=True)

    grid = defaultdict(lambda: defaultdict(list))       # tier -> (k,c) -> [rel_stem]
    grid_all = defaultdict(lambda: defaultdict(list))
    refs = defaultdict(list)
    ranked_cache = {}
    ratio_stats = defaultdict(lambda: defaultdict(list))  # tier -> ratio -> [(keep, ratio, fb)]
    per_q = []

    t0 = time.time()
    for i, q in enumerate(questions, 1):
        domain = q["domain"]
        options = q.get("options") or {}
        doc_ids = q.get("doc_ids") or domain_doc_ids(domain)
        feats = extract_question_features(q["question"], options, qid=q["qid"], domain=domain,
                                          doc_ids=doc_ids, answer_format=q.get("answer_format"),
                                          q_type=q.get("type"))
        spans_stem = [s for s in feats.key_spans if s.is_strong and s.text in q["question"]]
        spans_all = [s for s in feats.key_spans if s.is_strong]
        if not spans_stem:
            spans_stem = spans_all
        pipe = DynamicMemoryPipeline(domain, doc_ids)
        ranked, idf, query = rank_chunks(pipe, q["question"], options)
        if not ranked or not spans_all:
            continue
        ranked_cache[q["qid"]] = (ranked, idf, query, spans_stem, spans_all,
                                  feats.tier, feats.is_calc)

        ref_stem = coverage_for(ranked, spans_stem, REFERENCE_TOP_K, None)
        ref_all = coverage_for(ranked, spans_all, REFERENCE_TOP_K, None)
        refs[feats.tier].append(ref_stem)
        for k in TOP_K_GRID:
            for c in CHAR_GRID:
                cov_s = coverage_for(ranked, spans_stem, k, c)
                cov_a = coverage_for(ranked, spans_all, k, c)
                grid[feats.tier][(k, c)].append(cov_s / ref_stem if ref_stem > 0 else 1.0)
                grid_all[feats.tier][(k, c)].append(cov_a / ref_all if ref_all > 0 else 1.0)
        per_q.append({"qid": q["qid"], "tier": feats.tier, "ref_stem_coverage": round(ref_stem, 4),
                      "n_strong_spans": len(spans_all)})
        if i % 10 == 0:
            print(f"  ...{i}/{len(questions)}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    rows = []
    pooled_rel: Dict[tuple, float] = {}
    for kc in list(grid[[t for t in grid][0]].keys()):
        vals = [v for tier in grid for v in grid[tier].get(kc, [])]
        if vals:
            pooled_rel[kc] = sum(vals) / len(vals)
    anchor_k, anchor_c, anchor_rel = choose_anchor(pooled_rel)
    ladder = build_tier_ladder(anchor_k, anchor_c)
    print(f"全题池膝点(锚): top_k={anchor_k}, max_chars={anchor_c}, "
          f"相对覆盖率={anchor_rel:.3f}; 阶梯={ladder}", flush=True)

    chosen: Dict[str, Dict[str, float]] = {}
    for tier in sorted(grid.keys()):
        mean_rel = {kc: sum(v) / len(v) for kc, v in grid[tier].items()}
        mean_rel_all = {kc: sum(v) / len(v) for kc, v in grid_all[tier].items()}
        ref_mean = sum(refs[tier]) / len(refs[tier]) if refs[tier] else 0.0
        for (k, c) in sorted(mean_rel.keys(), key=lambda x: (x[1], x[0])):
            rows.append({
                "split": split, "tier": tier, "top_k": k, "max_chars": c,
                "n_questions": len(grid[tier][(k, c)]),
                "mean_relative_coverage_stem": round(mean_rel[(k, c)], 4),
                "mean_relative_coverage_stem_options": round(mean_rel_all[(k, c)], 4),
                "reference_coverage_stem": round(ref_mean, 4),
                "mean_relative_coverage_pooled": round(pooled_rel[(k, c)], 4),
            })
        k, c = ladder[tier]["top_k"], ladder[tier]["max_chars"]
        chosen[tier] = {
            "top_k": k, "max_chars": c,
            "coverage_at_choice": round(mean_rel[(k, c)], 4),
            "pooled_coverage_at_choice": round(pooled_rel[(k, c)], 4),
            "reference_coverage_stem": round(ref_mean, 4),
            "n_questions": len(grid[tier][(k, c)]),
        }

    grid_path = os.path.join(config.EXPERIMENTS_DIR, f"budget_tuning_{split.lower()}.csv")
    with open(grid_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"网格结果: {grid_path}", flush=True)

    # ---- 压缩比扫描（用已缓存排序结果，不重复检索）----
    chars_stat: Dict[str, List[int]] = defaultdict(list)
    tok_stat: Dict[str, List[int]] = defaultdict(list)
    estimator = load_estimator()
    for qid, (ranked, idf, query, spans_stem, spans_all, tier, is_calc) in ranked_cache.items():
        params = chosen[tier]
        picked, used = [], 0
        for ev in ranked[:params["top_k"]]:
            if used + len(ev.text) > params["max_chars"] and picked:
                continue
            picked.append(ev)
            used += len(ev.text)
        if not picked:
            continue
        chars_stat[tier].append(used)
        tok_stat[tier].append(estimator.estimate("\n".join(ev.text for ev in picked)))
        for r in RATIO_GRID:
            res = compress_evidence(picked, spans_all, compression_ratio=r,
                                    query_tokens=tokenize(query, use_finance_dict=True),
                                    idf=idf, finance_terms=FINANCE_TERMS,
                                    verbatim_top_n=2 if is_calc else 0)
            ratio_stats[tier][r].append((res.protected_keep_rate, res.compression_ratio,
                                         res.fallback_chunks / max(1, res.n_chunks_before),
                                         res.n_sentences_after / max(1, res.n_sentences_before)))

    comp_rows = []
    ratio_choice: Dict[str, float] = {}
    for tier in sorted(ratio_stats.keys()):
        pick = None
        for r in sorted(RATIO_GRID):
            vals = ratio_stats[tier].get(r) or []
            if not vals:
                continue
            keep = sum(v[0] for v in vals) / len(vals)
            real = sum(v[1] for v in vals) / len(vals)
            fb = sum(v[2] for v in vals) / len(vals)
            sent = sum(v[3] for v in vals) / len(vals)
            comp_rows.append({"split": split, "tier": tier, "target_ratio": r,
                              "mean_protected_keep_rate": round(keep, 4),
                              "mean_realized_ratio": round(real, 4),
                              "mean_fallback_share": round(fb, 4),
                              "mean_sentence_keep_share": round(sent, 4),
                              "n_questions": len(vals)})
            # 验收标准：关键跨度 100% 保留 + 回退块不超过一半 + 至少保留 40% 的句子
            # （最后一条避免"数字都在、但支撑推理的句子被删光"）
            if pick is None and keep >= 0.999 and fb <= 0.5 and sent >= 0.4:
                pick = r
        ratio_choice[tier] = pick if pick is not None else 0.5

    comp_path = os.path.join(config.EXPERIMENTS_DIR, f"compression_tuning_{split.lower()}.csv")
    if comp_rows:
        with open(comp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
            w.writeheader()
            w.writerows(comp_rows)
        print(f"压缩扫描: {comp_path}", flush=True)

    tiers = {}
    for tier, params in chosen.items():
        tiers[tier] = {
            "top_k": int(params["top_k"]),
            "max_chars": int(params["max_chars"]),
            "compression_ratio": ratio_choice.get(tier, 0.5),
            "mean_relative_coverage_stem": params["coverage_at_choice"],
            "pooled_relative_coverage": params["pooled_coverage_at_choice"],
            "reference_coverage_stem": params.get("reference_coverage_stem"),
            "n_questions": params["n_questions"],
            "mean_context_chars_before_compression": (
                round(sum(chars_stat[tier]) / len(chars_stat[tier]), 1) if chars_stat[tier] else None),
            "mean_estimated_prompt_tokens": (
                round(sum(tok_stat[tier]) / len(tok_stat[tier]), 1) if tok_stat[tier] else None),
        }
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split,
        "objective": (f"全题池覆盖率曲线膝点（相对覆盖率≥{COVERAGE_FLOOR} 的最小 (max_chars, top_k)），"
                      f"三档按 0.5x/1x/1.5x 缩放"),
        "reference": f"top_k={REFERENCE_TOP_K}, 无字符上限",
        "coverage_floor": COVERAGE_FLOOR,
        "anchor": {"top_k": anchor_k, "max_chars": anchor_c,
                   "pooled_relative_coverage": round(anchor_rel, 4)},
        "note": "无 gold label 下的代理目标；拿到标注后请用 experiments/run_ablation.py 复核",
        "tiers": tiers,
    }
    out_json = config.BUDGET_TIERS_FILE if split == "A" else os.path.join(
        config.EXPERIMENTS_DIR, "budget_tuning_b.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    per_q_path = os.path.join(config.EXPERIMENTS_DIR, f"budget_tuning_{split.lower()}_per_question.json")
    with open(per_q_path, "w", encoding="utf-8") as f:
        json.dump(per_q, f, ensure_ascii=False, indent=2)
    print(f"三档参数已写入: {out_json}", flush=True)
    print(json.dumps(tiers, ensure_ascii=False, indent=2), flush=True)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="A", choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    tune(args.split, args.limit)
    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
