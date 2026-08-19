"""评估脚本：在标注了标准答案的题集上计算系统准确率。

用法：
  1. 在 eval_labels.json 中填写标准答案（qid -> 答案字符串）
  2. python evaluate.py              # 跑全部已标注题目
     python evaluate.py --limit 5    # 只跑前 5 道（省 token）

输出：逐题命中情况，以及按领域 / 按题型的准确率汇总。
"""
import argparse
import json
import os

from main import load_questions, solve_question

LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_labels.json")


def normalize_answer(ans: str) -> str:
    """统一答案格式：只保留 A-D，去重并按字母序排列（与评测规则一致）。"""
    letters = "".join(sorted(set(c for c in ans.upper() if c in "ABCD")))
    return letters


def load_labels(path: str = LABELS_PATH) -> dict:
    """读取标注文件，兼容 {"qid": "AC"} 或 {"labels": {"qid": "AC"}}。"""
    if not os.path.exists(path):
        print(f"[WARN] 未找到标注文件: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", data) if isinstance(data, dict) else {}
    cleaned = {}
    for qid, ans in labels.items():
        norm = normalize_answer(str(ans))
        if norm:
            cleaned[qid] = norm
    return cleaned


def evaluate(labels: dict, limit: int = None) -> None:
    """在已标注题目上运行完整作答流程并输出准确率汇总。"""
    questions = {q["qid"]: q for q in load_questions("A")}

    labeled = [qid for qid in labels if qid in questions]
    missing = [qid for qid in labels if qid not in questions]
    if missing:
        print(f"[WARN] 以下标注题目不在 A 榜数据中，已跳过: {missing}")

    if limit:
        labeled = labeled[:limit]

    if not labeled:
        print("[INFO] 没有可评估的题目，请在 eval_labels.json 中填写标准答案。")
        return

    results = []
    for qid in labeled:
        q = questions[qid]
        r = solve_question(q)
        pred = normalize_answer(r["answer"])
        truth = labels[qid]
        hit = pred == truth
        results.append({
            "qid": qid,
            "domain": q["domain"],
            "answer_format": q["answer_format"],
            "truth": truth,
            "pred": pred or "(空)",
            "hit": hit,
        })
        mark = "OK" if hit else "XX"
        print(f"  [{mark}] {qid}: 预测={pred or '(空)'} 标准={truth}")

    total = len(results)
    correct = sum(1 for r in results if r["hit"])
    print(f"\n准确率: {correct}/{total} = {correct / total:.1%}")

    for key, label in (("domain", "领域"), ("answer_format", "题型")):
        groups = {}
        for r in results:
            groups.setdefault(r[key], []).append(r)
        print(f"\n按{label}拆分:")
        for name, rs in sorted(groups.items()):
            c = sum(1 for r in rs if r["hit"])
            print(f"  [{name}] {c}/{len(rs)} = {c / len(rs):.1%}")


def main():
    parser = argparse.ArgumentParser(description="金融长文档 QA 评估脚本")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 道已标注题目（省 token）")
    args = parser.parse_args()

    labels = load_labels()
    if not labels:
        print('请先在 eval_labels.json 中填写标准答案，例如 {"reg_a_014": "AC"}')
        return
    evaluate(labels, limit=args.limit)


if __name__ == "__main__":
    main()
