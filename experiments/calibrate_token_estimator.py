"""用真实 API 返回的 prompt_tokens 标定 token 估计器（离线，不调用 API）。

思路：
  A 榜那次真实运行（run_log.txt）里每道题都记录了
    - "检索完成: N chars"     → 上下文长度
    - "tokens: PT+CT"          → 该题真实 prompt/completion token
  并且已经验证：用 `retrieve_context(..., max_chars=12000)` 可以**逐题精确复现**
  当时的上下文文本（例如 fc_a_001=11845、fc_a_003=11936）。
  因此可以重建当时发给 Qwen 的完整 prompt，并用真实 prompt_tokens 做标定。

标定模型：prompt_tokens ≈ a*中文字符数 + b*ASCII字母数字数 + c*其它字符数 + d

验证方式：
  * 单次调用题（单选/判断，且未触发二次回退）直接作为标签，报告 MAPE / R²；
  * 多选题：用估计器反推"该题调用了几次"，检查是否落在 {2,3}（真实逻辑上限），
    并报告与整数的平均偏差。

产出：experiments/token_estimator.json、experiments/token_estimator_validation.csv
"""
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.token_estimator import (TokenEstimator, fit_estimator, save_estimator,
                                   select_and_fit)
from reasoner import SYSTEM_PROMPT, build_user_prompt
from retriever import retrieve_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_MAX_CHARS = 12000        # 由 run_log.txt 逐题核对得到的历史运行上限


def read_text_any(path: str) -> str:
    raw = open(path, "rb").read()
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_run_log(path: str) -> List[Dict[str, object]]:
    """从运行日志里抽出 每题 → (上下文chars, prompt_tokens, completion_tokens)。"""
    text = read_text_any(path)
    out: List[Dict[str, object]] = []
    cur: Optional[str] = None
    period: Optional[str] = None
    chars: Optional[int] = None
    for line in text.split("\n"):
        m = re.search(r"\[(\w+_[ab]_\d+)\]\s*\(([\w/]+)\)", line)
        if m:
            cur, period = m.group(1), m.group(2)
            chars = None
            continue
        m = re.search(r"检索完成:\s*(\d+)\s*chars", line)
        if m and cur:
            chars = int(m.group(1))
            continue
        m = re.search(r"tokens:\s*(\d+)\+(\d+)=(\d+)", line)
        if m and cur:
            out.append({"qid": cur, "period": period, "context_chars": chars,
                        "prompt_tokens": int(m.group(1)),
                        "completion_tokens": int(m.group(2))})
            cur = None
    return out


def load_questions() -> Dict[str, dict]:
    qs: Dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(config.QUESTIONS_DIR, "*.json"))):
        for q in json.load(open(p, encoding="utf-8")):
            qs[q["qid"]] = q
    return qs


def rebuild_prompt(q: dict, context: str) -> str:
    """重建当时发给模型的完整 prompt（system + user）。"""
    user = build_user_prompt(q["question"], q["options"], q["answer_format"], context)
    return SYSTEM_PROMPT + "\n" + user


def calibrate(log_path: str = "run_log.txt") -> Dict[str, object]:
    rows = parse_run_log(os.path.join(ROOT, log_path))
    qs = load_questions()
    print(f"日志解析到 {len(rows)} 题")

    samples: List[Dict[str, object]] = []
    for r in rows:
        q = qs.get(str(r["qid"]))
        if not q:
            continue
        ctx = retrieve_context(q["doc_ids"], q["domain"], q["question"], q["options"],
                               max_chars=LEGACY_MAX_CHARS)
        if r["context_chars"] is not None and len(ctx) != r["context_chars"]:
            print(f"  [WARN] {r['qid']} 上下文长度不一致: 复现={len(ctx)} 日志={r['context_chars']}")
        r["reproduced_chars"] = len(ctx)
        r["prompt_text"] = rebuild_prompt(q, ctx)
        r["answer_format"] = q["answer_format"]
        samples.append(r)
    print(f"可复现样本 {len(samples)} 题")

    # ---- 用单次调用题拟合 ----
    singles = [s for s in samples if s["answer_format"] in ("mcq", "tf")]
    print(f"单次调用样本(拟合用) {len(singles)} 题")
    est = select_and_fit([s["prompt_text"] for s in singles],
                         [s["prompt_tokens"] for s in singles])
    # 二次稳健化：第一次拟合残差 >25% 的样本（可能触发了 fallback 二次调用）剔除后重拟合
    keep = []
    for s in singles:
        pred = est.estimate(s["prompt_text"])
        if s["prompt_tokens"] and abs(pred - s["prompt_tokens"]) / s["prompt_tokens"] <= 0.25:
            keep.append(s)
    if 5 <= len(keep) < len(singles):
        est = select_and_fit([s["prompt_text"] for s in keep], [s["prompt_tokens"] for s in keep])
        print(f"稳健化后保留 {len(keep)}/{len(singles)} 个单次调用样本")

    val_rows = []
    abs_err = []
    rel_err = []
    for s in samples:
        pred_call = est.estimate(s["prompt_text"])
        ratio = (s["prompt_tokens"] / pred_call) if pred_call else 0.0
        val_rows.append({
            "qid": s["qid"], "answer_format": s["answer_format"],
            "context_chars": s["context_chars"], "reproduced_chars": s["reproduced_chars"],
            "observed_prompt_tokens": s["prompt_tokens"],
            "estimated_single_call_tokens": pred_call,
            "implied_calls": round(ratio, 2),
            "in_single_call_fit": int(s in keep),
        })
        if s in keep:
            abs_err.append(abs(pred_call - s["prompt_tokens"]))
            rel_err.append(abs(pred_call - s["prompt_tokens"]) / s["prompt_tokens"])

    implied = [r["implied_calls"] for r in val_rows if r["answer_format"] == "multi"]
    int_dev = [abs(x - round(x)) for x in implied]
    within = [1 if 1.5 <= x <= 3.5 else 0 for x in implied]
    extra = {
        "calibration_source": log_path,
        "legacy_max_chars": LEGACY_MAX_CHARS,
        "n_single_call_samples": len(keep),
        "single_call_mape_percent": round(100.0 * sum(rel_err) / len(rel_err), 3) if rel_err else None,
        "single_call_mean_abs_error": round(sum(abs_err) / len(abs_err), 1) if abs_err else None,
        "multi_questions": len(implied),
        "multi_implied_calls_in_2_3_ratio": round(sum(within) / len(within), 4) if within else None,
        "multi_mean_integer_deviation": round(sum(int_dev) / len(int_dev), 4) if int_dev else None,
    }
    est.metrics.update(extra)

    out = save_estimator(est, extra=extra)
    print(f"估计器已保存: {out}")
    print(json.dumps({"coef": est.coef, "metrics": est.metrics}, ensure_ascii=False, indent=2))

    csv_path = os.path.join(config.EXPERIMENTS_DIR, "token_estimator_validation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        w.writeheader()
        w.writerows(val_rows)
    print(f"逐题验证结果: {csv_path}")
    return est.metrics


if __name__ == "__main__":
    calibrate()
