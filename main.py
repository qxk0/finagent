"""主流程 - 加载题目 → 检索 → 推理 → 生成 answer.csv"""
import os
import sys
import json
import csv
import time
from typing import List

import config
from preprocess import preprocess_all, load_document
from retriever import retrieve_context
from reasoner import answer_question


def load_questions(split: str = "A") -> List[dict]:
    """加载所有题目"""
    questions = []
    q_dir = config.QUESTIONS_DIR
    if not os.path.exists(q_dir):
        print(f"[ERROR] 题目目录不存在: {q_dir}")
        return []

    for fname in sorted(os.listdir(q_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(q_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for q in data:
            if q.get("split", "A") == split:
                questions.append(q)

    print(f"加载 {len(questions)} 道 {split} 组题目")
    return questions


def solve_question(q: dict) -> dict:
    """解答单道题目"""
    qid = q["qid"]
    domain = q["domain"]
    question = q["question"]
    options = q["options"]
    answer_format = q["answer_format"]
    doc_ids = q.get("doc_ids", [])

    print(f"\n{'='*60}")
    print(f"[{qid}] ({domain}/{answer_format}) {question[:50]}...")

    # 1. 检索证据
    t0 = time.time()
    if doc_ids:
        context = retrieve_context(doc_ids, domain, question, options)
    else:
        # B组无doc_ids，需要全领域检索（后续扩展）
        context = "[无指定文档，需全领域检索]"

    t1 = time.time()
    print(f"  检索完成: {len(context)} chars, {t1-t0:.1f}s")

    if not context or context.startswith("[无"):
        print(f"  [WARN] 无有效上下文")
        return {
            "qid": qid,
            "answer": "A",  # 默认答案
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # 2. 推理作答
    answer, pt, ct = answer_question(question, options, answer_format, context)
    t2 = time.time()
    print(f"  答案: {answer} | tokens: {pt}+{ct}={pt+ct} | {t2-t1:.1f}s")

    return {
        "qid": qid,
        "answer": answer,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
    }


def generate_csv(results: List[dict], output_path: str = None):
    """生成提交用 answer.csv"""
    if output_path is None:
        output_path = config.OUTPUT_CSV

    total_pt = sum(r["prompt_tokens"] for r in results)
    total_ct = sum(r["completion_tokens"] for r in results)
    total_tokens = total_pt + total_ct

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        # summary 行
        writer.writerow(["summary", "", total_pt, total_ct, total_tokens])
        # 每题
        for r in results:
            writer.writerow([r["qid"], r["answer"], r["prompt_tokens"],
                           r["completion_tokens"], r["total_tokens"]])

    print(f"\n{'='*60}")
    print(f"CSV 已生成: {output_path}")
    print(f"总 Token: {total_tokens:,} (prompt={total_pt:,}, completion={total_ct:,})")
    token_score = max(0, min(1, (5_000_000 - total_tokens) / 5_000_000))
    print(f"TokenScore: {token_score:.4f}")
    print(f"预估得分系数: {0.7 + 0.3 * token_score:.4f}")


def main():
    # 检查 API Key
    if not config.DASHSCOPE_API_KEY:
        print("[ERROR] 请设置环境变量 DASHSCOPE_API_KEY")
        print("  Windows: set DASHSCOPE_API_KEY=sk-xxx")
        print("  Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx")
        sys.exit(1)

    print("=" * 60)
    print("金融长文档 QA Agent 启动")
    print(f"模型: {config.MODEL_NAME}")
    print(f"数据: {config.DATA_DIR}")
    print("=" * 60)

    # 1. 加载题目
    questions = load_questions("A")
    if not questions:
        print("[ERROR] 未找到题目")
        sys.exit(1)

    # 2. 预处理文档（缓存）
    print("\n--- 文档预处理 ---")
    preprocess_all(questions)

    # 3. 逐题作答
    print("\n--- 开始作答 ---")
    results = []
    for i, q in enumerate(questions):
        print(f"\n进度: {i+1}/{len(questions)}")
        result = solve_question(q)
        results.append(result)

    # 4. 生成 CSV
    generate_csv(results)

    # 5. 打印汇总
    print("\n--- 作答汇总 ---")
    for r in results:
        print(f"  {r['qid']}: {r['answer']} ({r['total_tokens']} tokens)")


if __name__ == "__main__":
    main()
