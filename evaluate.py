"""Golden Set 评测脚本（任务3）。

能力：
  1. 标准答案来源审计（数据里到底有没有 gold label）——`--audit`
  2. 自动构建 eval_labels.json（仅当数据中真实存在答案时）——`--build-labels`
  3. 用 gold label 评测系统：overall / domain / answer_format / 单选·多选·判断
     四类准确率 + 混淆矩阵 + 失败清单
  4. 也可以直接对已有提交文件（answer.csv / answer_b.csv）离线打分，不消耗 token

用法：
  python evaluate.py --audit                     # 先确认有没有标准答案
  python evaluate.py                             # 跑 A 榜并打分（需 API Key）
  python evaluate.py --limit 10                  # 只跑 10 题（省 token）
  python evaluate.py --predictions answer.csv    # 只对已有提交打分

输出：控制台报告 + experiments/eval_report.json

截至当前版本，数据集中**不存在任何标准答案**（见 experiments/labels_audit.json），
因此本脚本在没有人工标注前无法给出真实准确率——它会明确报告缺失，而不会编造数字。
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(ROOT, "eval_labels.json")
TEMPLATE_PATH = os.path.join(config.EXPERIMENTS_DIR, "eval_labels_template.json")
REPORT_PATH = os.path.join(config.EXPERIMENTS_DIR, "eval_report.json")

CHOICE_FORMATS = ("mcq", "multi", "tf")


def repo_rel(path: str) -> str:
    """报告里只记录仓库相对路径，避免本机绝对路径进入公开仓库。"""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.basename(path)


# ------------------------------------------------------------------ 归一化

def normalize_answer(ans: str, answer_format: Optional[str] = None) -> str:
    """与官方评测口径一致的答案归一化（选择/判断题按字母集合比较）。

    * 选择/判断题：只保留 A-D 字母，去重并按字母序排序（ACD == CAD）
    * 计算/抽取题：去空白、统一全角字符、%、比较符；保留数字与单位
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    if not s:
        return ""
    fmt = (answer_format or "").lower()
    if fmt in CHOICE_FORMATS or re.fullmatch(r"[A-Da-d,，、;；\s]+", s):
        letters = "".join(sorted(set(c for c in s.upper() if c in "ABCD")))
        return letters
    s = s.replace("％", "%").replace("，", ",").replace("．", ".").replace("＞", ">")
    s = re.sub(r"\s+", "", s)
    s = s.strip("。.;；,，")
    return s


def is_choice_like(q: dict) -> bool:
    fmt = (q.get("answer_format") or "").lower()
    if fmt in CHOICE_FORMATS:
        return True
    if fmt:
        return False
    tpe = q.get("type") or ""
    return any(k in tpe for k in ("单选", "多选", "判断")) or bool(q.get("options"))


def question_kind(q: dict) -> str:
    """single / multi / tf / calc / extract"""
    fmt = (q.get("answer_format") or "").lower()
    tpe = q.get("type") or ""
    if fmt == "multi" or "多选" in tpe:
        return "multi"
    if fmt == "tf" or "判断" in tpe:
        return "tf"
    if fmt == "mcq" or "单选" in tpe:
        return "single"
    if "计算" in tpe:
        return "calc"
    if "抽取" in tpe:
        return "extract"
    return "single" if q.get("options") else "calc"


# ------------------------------------------------------------------ 标签

def load_labels(path: str = LABELS_PATH) -> Dict[str, str]:
    """读取标注文件，兼容 {"qid": "AC"} / {"labels": {...}}。"""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", data) if isinstance(data, dict) else {}
    out = {}
    for qid, ans in labels.items():
        if str(qid).startswith("_") or not str(ans).strip():
            continue
        out[str(qid)] = str(ans).strip()
    return out


def load_questions(split: str = "A") -> List[dict]:
    if split == "A":
        out = []
        for fname in sorted(os.listdir(config.QUESTIONS_DIR)):
            if fname.endswith(".json"):
                with open(os.path.join(config.QUESTIONS_DIR, fname), "r", encoding="utf-8") as f:
                    out += json.load(f)
        return out
    out = []
    b_dir = os.path.join(ROOT, "upload_b", "question_b")
    if not os.path.isdir(b_dir):
        return out
    for fname in sorted(os.listdir(b_dir)):
        p = os.path.join(b_dir, fname)
        if fname.endswith(".jsonl"):
            with open(p, "r", encoding="utf-8-sig") as f:
                out += [json.loads(l) for l in f if l.strip()]
        elif fname.endswith(".json"):
            with open(p, "r", encoding="utf-8-sig") as f:
                out += json.load(f)
    return out


def load_predictions(path: str) -> Dict[str, str]:
    """从 answer.csv / answer_b.csv 读取预测（answer_1..4 拼接）。"""
    preds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            if not qid or qid == "summary":
                continue
            if row.get("answer") is not None:
                preds[qid] = (row.get("answer") or "").strip()
            else:
                parts = [(row.get(f"answer_{i}") or "").strip() for i in range(1, 5)]
                joined = "".join(p for p in parts if p)
                preds[qid] = parts[0] if len(parts) > 1 and not all(
                    re.fullmatch(r"[A-D]{1,4}", p or "") for p in parts if p) else joined
    return preds


# ------------------------------------------------------------------ 评测

def score(labels: Dict[str, str], predictions: Dict[str, str],
          questions: Dict[str, dict]) -> Dict[str, object]:
    per_q = []
    for qid, truth_raw in labels.items():
        q = questions.get(qid)
        if not q:
            continue
        fmt = q.get("answer_format") if q.get("answer_format") else q.get("type")
        truth = normalize_answer(truth_raw, q.get("answer_format"))
        pred_raw = predictions.get(qid, "")
        pred = normalize_answer(pred_raw, q.get("answer_format"))
        per_q.append({
            "qid": qid,
            "domain": q.get("domain", ""),
            "answer_format": q.get("answer_format") or q.get("type") or "",
            "kind": question_kind(q),
            "truth": truth,
            "pred": pred or "(空)",
            "pred_raw": pred_raw,
            "hit": bool(truth) and pred == truth,
        })

    n = len(per_q)
    if n == 0:
        return {"n": 0, "note": "没有可评测的题目（标签 qid 与题集不匹配）"}

    def acc(rows):
        return round(sum(1 for r in rows if r["hit"]) / len(rows), 4) if rows else None

    by_domain = defaultdict(list)
    by_format = defaultdict(list)
    by_kind = defaultdict(list)
    for r in per_q:
        by_domain[r["domain"]].append(r)
        by_format[r["answer_format"]].append(r)
        by_kind[r["kind"]].append(r)

    choice_rows = [r for r in per_q if re.fullmatch(r"[A-D]*", r["truth"]) and r["truth"]]
    confusion = sorted(Counter((r["truth"], r["pred"]) for r in choice_rows).items(),
                       key=lambda x: -x[1])
    failures = [
        {"qid": r["qid"], "domain": r["domain"], "kind": r["kind"],
         "truth": r["truth"], "pred": r["pred"], "pred_raw": r["pred_raw"]}
        for r in per_q if not r["hit"]
    ]
    failure_by_kind = Counter(f["kind"] for f in failures)
    failure_by_domain = Counter(f["domain"] for f in failures)

    return {
        "n": n,
        "overall_accuracy": acc(per_q),
        "domain_accuracy": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_domain.items())},
        "answer_format_accuracy": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_format.items())},
        "kind_accuracy": {k: {"n": len(v), "accuracy": acc(v)} for k, v in sorted(by_kind.items())},
        "single_multi_tf_accuracy": {
            "single": acc(by_kind.get("single", [])),
            "multi": acc(by_kind.get("multi", [])),
            "tf": acc(by_kind.get("tf", [])),
            "calc": acc(by_kind.get("calc", [])),
            "extract": acc(by_kind.get("extract", [])),
        },
        "confusion_top": [{"truth": t, "pred": p, "count": c} for (t, p), c in confusion[:30]],
        "failure_summary": {
            "n_failures": len(failures),
            "by_kind": dict(failure_by_kind),
            "by_domain": dict(failure_by_domain),
            "failures": failures,
        },
        "per_question": per_q,
    }


def print_report(report: Dict[str, object]) -> None:
    if not report.get("n"):
        print(f"[INFO] {report.get('note')}")
        return
    print(f"\n总题数: {report['n']}")
    print(f"总体准确率: {report['overall_accuracy']:.2%}")
    print("\n按领域:")
    for k, v in report["domain_accuracy"].items():
        print(f"  [{k:<22}] {v['n']:>3} 题  准确率 {v['accuracy']:.2%}")
    print("\n按题型(answer_format):")
    for k, v in report["answer_format_accuracy"].items():
        print(f"  [{k:<10}] {v['n']:>3} 题  准确率 {v['accuracy']:.2%}")
    print("\n按题目类型(single/multi/tf/calc/extract):")
    for k, v in report["single_multi_tf_accuracy"].items():
        if v is not None:
            print(f"  [{k:<8}] 准确率 {v:.2%}")
    print("\n混淆矩阵 Top（truth → pred）:")
    for row in report["confusion_top"][:15]:
        print(f"  {row['truth']:>5} → {row['pred']:<5} : {row['count']}")
    fs = report["failure_summary"]
    print(f"\n失败题数: {fs['n_failures']}  按类型 {fs['by_kind']}  按领域 {fs['by_domain']}")
    for f in fs["failures"][:20]:
        print(f"  XX {f['qid']} ({f['domain']}/{f['kind']}) 预测={f['pred']} 标准={f['truth']}")
    if fs["n_failures"] > 20:
        print(f"  ... 其余 {fs['n_failures'] - 20} 条见 {REPORT_PATH}")


# ------------------------------------------------------------------ 主流程

def run_audit() -> Dict[str, object]:
    """调用 build_eval_labels 的审计逻辑（不生成标签，除非有真实答案）。"""
    sys.argv = [sys.argv[0]]
    import experiments.build_eval_labels as bel
    recs = bel.iter_question_records()
    field_scan = bel.scan_question_fields(recs)
    sidecar = bel.scan_sidecar_files()
    audit = {"question_field_scan": field_scan,
             "sidecar": sidecar,
             "labels_json_exists": os.path.exists(LABELS_PATH),
             "labels_json_nonempty": bool(load_labels())}
    return audit


def main():
    ap = argparse.ArgumentParser(description="金融长文档 QA - Golden Set 评测")
    ap.add_argument("--labels", default=LABELS_PATH)
    ap.add_argument("--split", default="A", choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 题（省 token）")
    ap.add_argument("--predictions", default=None, help="直接对已有提交文件打分，不调用 API")
    ap.add_argument("--audit", action="store_true", help="只做标准答案存在性审计")
    ap.add_argument("--build-labels", action="store_true", help="尝试自动构建 eval_labels.json")
    ap.add_argument("--report", default=REPORT_PATH, help="评测报告输出路径")
    args = ap.parse_args()

    if args.build_labels:
        import experiments.build_eval_labels as bel
        sys.argv = [sys.argv[0]]
        bel.main()
        return

    if args.audit:
        audit = run_audit()
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        print("\n结论: " + ("数据中存在候选答案来源，请人工确认后运行 --build-labels --force-source <file>"
                          if any(s.get("verdict") == "CANDIDATE" for s in audit["sidecar"]["sources"])
                          else "数据中未发现标准答案；无法计算真实准确率（不猜测标签）"))
        return

    labels = load_labels(args.labels)
    if not labels:
        print("=" * 70)
        print("[无法评测] 没有标准答案。")
        print(f"  - {args.labels} 为空或不存在")
        print(f"  - 题目文件里只有 qid/domain/split/question/options/type/answer_format/doc_ids，"
              f"没有任何答案字段")
        print(f"  - 数据目录里也没有带真实答案的文件（submit.csv 为占位值 999999.99%）")
        print(f"  - 审计详情: python evaluate.py --audit")
        print(f"  - 人工标注模板: {TEMPLATE_PATH}（填好后用 --labels 指定）")
        report = {"n": 0, "status": "NO_GOLD_LABELS", "labels_path": repo_rel(args.labels)}
        os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return

    questions = {q["qid"]: q for q in load_questions(args.split)}
    label_qids = [q for q in labels if q in questions]
    missing = [q for q in labels if q not in questions]
    if missing:
        print(f"[WARN] {len(missing)} 条标签不在 {args.split} 榜题集中: {missing[:5]}...")
    if args.limit:
        label_qids = label_qids[:args.limit]

    if args.predictions:
        preds = load_predictions(args.predictions)
        print(f"从 {args.predictions} 读取 {len(preds)} 条预测（不调用 API）")
    else:
        from reasoner import answer_question, has_api_key
        if not has_api_key():
            print("[ERROR] 未设置 DASHSCOPE_API_KEY；如需只对已有提交打分，请加 --predictions answer.csv")
            return
        preds = {}
        for i, qid in enumerate(label_qids, 1):
            q = questions[qid]
            from retriever import retrieve_context
            context = retrieve_context(q.get("doc_ids", []), q["domain"], q["question"], q.get("options", {}))
            ans, pt, ct = answer_question(q["question"], q.get("options", {}),
                                          q.get("answer_format", "mcq"), context, qid=qid)
            preds[qid] = ans
            print(f"  [{i}/{len(label_qids)}] {qid}: {ans} ({pt + ct} tokens)")

    report = score({q: labels[q] for q in label_qids}, preds, questions)
    print_report(report)
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n完整报告: {args.report}")


if __name__ == "__main__":
    main()
