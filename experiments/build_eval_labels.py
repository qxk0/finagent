"""Golden Set 构建 / 缺失审计（任务3）。

规则（严格遵守"不猜答案"）：
  * 只在数据文件里**真实存在**标准答案字段/文件时，才自动生成 eval_labels.json；
  * 任何情况下都不使用模型自己的输出（answer.csv / answer_b.csv）作为标签；
  * 检测到占位值（如 submit.csv 里的 999999.99%）时，标记为 PLACEHOLDER 并排除；
  * 找不到答案时输出明确的缺失报告 + 一个空的标注模板供人工填写。

产出：
  experiments/labels_audit.json          逐来源审计结果（含结论）
  eval_labels.json                       仅当发现可信标签时才写入
  experiments/eval_labels_template.json  人工标注模板（qid -> ""）
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_rel(path: str) -> str:
    """审计产物里只记录仓库相对路径，避免把本机绝对路径写进公开仓库。"""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.basename(path)

GOLD_FIELD_CANDIDATES = [
    "answer", "answers", "answer_1", "gold", "gold_answer", "label", "labels",
    "expected", "expected_answer", "correct", "correct_answer", "reference",
    "reference_answer", "ground_truth", "gt", "答案", "标准答案", "正确答案",
]
PLACEHOLDER_PATTERNS = [
    re.compile(r"^999999(\.99)?%?$"),
    re.compile(r"^x+$", re.I),
    re.compile(r"^(待填|未知|unknown|placeholder)$", re.I),
]


def iter_question_records() -> List[Dict[str, object]]:
    recs: List[Dict[str, object]] = []
    for p in sorted(glob.glob(os.path.join(config.QUESTIONS_DIR, "*.json"))):
        for q in json.load(open(p, encoding="utf-8")):
            recs.append({"qid": q["qid"], "split": q.get("split", "A"), "file": p, "record": q})
    b_dir = os.path.join(ROOT, "upload_b", "question_b")
    for p in sorted(glob.glob(os.path.join(b_dir, "*"))):
        if p.endswith(".jsonl"):
            for line in open(p, encoding="utf-8-sig"):
                if line.strip():
                    q = json.loads(line)
                    recs.append({"qid": q["qid"], "split": "B", "file": p, "record": q})
        elif p.endswith(".json"):
            for q in json.load(open(p, encoding="utf-8-sig")):
                recs.append({"qid": q["qid"], "split": "B", "file": p, "record": q})
    return recs


def is_placeholder(value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    return any(p.match(v) for p in PLACEHOLDER_PATTERNS)


def scan_question_fields(recs: List[Dict[str, object]]) -> Dict[str, object]:
    """题目 JSON 里是否直接带答案字段。"""
    found: Dict[str, int] = {}
    for r in recs:
        for key in GOLD_FIELD_CANDIDATES:
            if key in r["record"]:
                found[key] = found.get(key, 0) + 1
    return {"fields_found": found, "n_questions": len(recs),
            "has_gold_in_questions": bool(found)}


def scan_sidecar_files() -> Dict[str, object]:
    """扫描数据集目录里的 json/jsonl/csv，找 qid + 答案 结构。"""
    sources: List[Dict[str, object]] = []
    search_dirs = [config.DATA_DIR, os.path.join(ROOT, "upload_b"), ROOT]
    seen = set()
    for d in search_dirs:
        for pat in ("*.json", "*.jsonl", "*.csv"):
            for p in sorted(glob.glob(os.path.join(d, pat))):
                if p in seen or p.startswith(os.path.join(config.EXPERIMENTS_DIR)):
                    continue
                seen.add(p)
                info = inspect_file(p)
                if info:
                    sources.append(info)
    return {"sources": sources}


def inspect_file(path: str) -> Optional[Dict[str, object]]:
    name = os.path.basename(path)
    if name in ("answer.csv", "answer_b.csv", "eval_labels.json"):
        return None      # 自身输出，不作为标签来源
    try:
        if path.endswith(".csv"):
            with open(path, "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return None
            cols = list(rows[0].keys())
            if "qid" not in [c.lower() for c in cols]:
                return None
            answer_cols = [c for c in cols if re.search(r"answer|答案|gold|label", c, re.I)]
            if not answer_cols:
                return None
            values = []
            for r in rows:
                for c in answer_cols:
                    v = (r.get(c) or "").strip()
                    if v:
                        values.append(v)
            n_placeholder = sum(1 for v in values if is_placeholder(v))
            return {"path": repo_rel(path), "type": "csv", "qid_column": "qid",
                    "answer_columns": answer_cols, "n_rows": len(rows),
                    "n_non_empty_values": len(values),
                    "placeholder_ratio": round(n_placeholder / len(values), 4) if values else 1.0,
                    "verdict": "PLACEHOLDER_SUSPECT" if values and n_placeholder / len(values) > 0.3
                    else "CANDIDATE"}
        else:
            if path.endswith(".jsonl"):
                recs = [json.loads(l) for l in open(path, encoding="utf-8-sig") if l.strip()]
            else:
                recs = json.load(open(path, encoding="utf-8-sig"))
            if isinstance(recs, dict):
                # 形如 {"labels": {...}}
                for key in ("labels", "answers", "gold"):
                    if key in recs and isinstance(recs[key], dict) and recs[key]:
                        n = len(recs[key])
                        vals = [str(v) for v in recs[key].values()]
                        n_ph = sum(1 for v in vals if is_placeholder(v))
                        return {"path": repo_rel(path), "type": "json-dict", "key": key, "n_rows": n,
                                "placeholder_ratio": round(n_ph / n, 4),
                                "verdict": "PLACEHOLDER_SUSPECT" if n_ph / n > 0.3 else "CANDIDATE"}
                return None
            if not isinstance(recs, list) or not recs:
                return None
            keys = set(recs[0].keys()) if isinstance(recs[0], dict) else set()
            hits = keys & set(GOLD_FIELD_CANDIDATES)
            if "qid" in keys and hits:
                return {"path": repo_rel(path), "type": "json-list", "answer_fields": sorted(hits),
                        "n_rows": len(recs), "verdict": "CANDIDATE"}
    except Exception as e:
        return {"path": repo_rel(path), "error": str(e), "verdict": "UNREADABLE"}
    return None


def main():
    ap = argparse.ArgumentParser(description="检查并构建 Golden Set 标签")
    ap.add_argument("--out", default=os.path.join(ROOT, "eval_labels.json"))
    ap.add_argument("--force-source", default=None,
                    help="显式指定可信标签文件（CSV/JSON）以生成 eval_labels.json")
    args = ap.parse_args()

    recs = iter_question_records()
    q_field = scan_question_fields(recs)
    sidecar = scan_sidecar_files()

    audit = {
        "n_questions_total": len(recs),
        "n_questions_A": sum(1 for r in recs if r["split"] == "A"),
        "n_questions_B": sum(1 for r in recs if r["split"] == "B"),
        "question_field_scan": q_field,
        "sidecar_files": sidecar["sources"],
        "conclusion": None,
        "message": None,
    }

    candidates = [s for s in sidecar["sources"] if s.get("verdict") == "CANDIDATE"]
    if q_field["has_gold_in_questions"]:
        audit["conclusion"] = "GOLD_FOUND_IN_QUESTION_FILES"
    elif candidates:
        audit["conclusion"] = "GOLD_CANDIDATE_FOUND_IN_SIDECAR"
    else:
        audit["conclusion"] = "NO_GOLD_LABELS_FOUND"
        audit["message"] = ("数据集中未发现任何标准答案字段或答案文件；"
                            "eval_labels.json 保持为空，不做任何猜测。")

    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    audit_path = os.path.join(config.EXPERIMENTS_DIR, "labels_audit.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"\n审计结果: {audit_path}")

    # 人工标注模板（永远是空的，防止误当成标签使用）
    template = {"labels": {r["qid"]: "" for r in recs},
                "_note": "人工标注模板：请逐题填入标准答案后再用于 evaluate.py"}
    tpl_path = os.path.join(config.EXPERIMENTS_DIR, "eval_labels_template.json")
    with open(tpl_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"标注模板: {tpl_path}")

    # 只有显式指定可信来源时才写 eval_labels.json
    if args.force_source:
        labels = load_labels_from(args.force_source)
        if labels:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"labels": labels,
                           "_source": os.path.basename(args.force_source),
                           "_generated_by": "experiments/build_eval_labels.py"},
                          f, ensure_ascii=False, indent=2)
            print(f"已根据显式指定的来源写入 {args.out}: {len(labels)} 条标签")
        else:
            print(f"[WARN] 从 {args.force_source} 未解析到任何标签")


def load_labels_from(path: str) -> Dict[str, str]:
    """从 CSV/JSON 读取 qid → 答案。支持 answer_1..answer_4 拼成多选答案。"""
    labels: Dict[str, str] = {}
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                qid = (row.get("qid") or "").strip()
                if not qid or qid == "summary":
                    continue
                parts = [str(row.get(c) or "").strip() for c in
                         ("answer_1", "answer_2", "answer_3", "answer_4") if c in row]
                parts = [p for p in parts if p and not is_placeholder(p)]
                if parts:
                    labels[qid] = "".join(sorted(set("".join(parts)))) if all(
                        len(p) <= 4 and re.fullmatch(r"[A-D]+", p) for p in parts) else parts[0]
    else:
        data = json.load(open(path, encoding="utf-8-sig"))
        payload = data.get("labels", data) if isinstance(data, dict) else {}
        for qid, ans in payload.items():
            if str(ans).strip() and not is_placeholder(str(ans)):
                labels[str(qid)] = str(ans).strip()
    return labels


if __name__ == "__main__":
    main()
