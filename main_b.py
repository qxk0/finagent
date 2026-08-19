"""B榜主流程 - 全领域检索 + 新题型(计算题/抽取题) + 新CSV格式"""
import os
import sys
import json
import csv
import re
import time
from typing import List, Tuple, Dict

import config
from preprocess import load_document, build_doc_chunks, chunk_text
from retriever import tokenize, BM25Retriever, build_query
from reasoner import _vote_multi

from rank_bm25 import BM25Okapi
from openai import OpenAI

# B榜计算题需要更多输出token
MAX_TOKENS_CALC = 1200
MAX_TOKENS_CHOICE = 500

_client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.BASE_URL,
)


def call_llm_b(system_prompt: str, user_prompt: str, max_tokens: int = 500,
               temperature: float = None) -> Tuple[str, int, int]:
    """B榜专用LLM调用，支持自定义max_tokens"""
    if temperature is None:
        temperature = config.TEMPERATURE
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=config.TIMEOUT,
            )
            text = resp.choices[0].message.content or ""
            pt = resp.usage.prompt_tokens if resp.usage else 0
            ct = resp.usage.completion_tokens if resp.usage else 0
            return text, pt, ct
        except Exception as e:
            print(f"  [RETRY {attempt+1}] API错误: {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return "", 0, 0


# ============ B榜配置 ============
B_QUESTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_b", "question_b")
B_OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer_b.csv")

# B榜全领域文档列表
DOMAIN_DOC_IDS = {
    "insurance": [str(i) for i in range(1, 17)],
    "financial_contracts": [f"text{i:02d}" for i in range(1, 15)],
    "financial_reports": [
        "annual_byd_2024_report", "annual_byd_2025_report",
        "annual_catl_2024_report", "annual_catl_2025_report",
        "annual_chinamobile_2025_report", "annual_cmb_2025_report",
        "annual_cscec_2024_report", "annual_cscec_2025_report",
        "annual_midea_2024_report", "annual_midea_2025_report",
    ],
    "research": [f"pack2_text{i:02d}" for i in range(1, 21)],
    "regulatory": [],  # 动态构建
}


def get_regulatory_doc_ids() -> List[str]:
    """获取regulatory领域doc_id（仅txt+html，跳过大型PDF附件避免OOM）"""
    raw = config.RAW_DIR
    doc_ids = []
    # txt files (6个主要法规文本)
    txt_dir = os.path.join(raw, "regulatory", "txt")
    if os.path.exists(txt_dir):
        for f in os.listdir(txt_dir):
            if f.endswith(".txt"):
                doc_ids.append(f[:-4])
    # html files (377个CSRC文档)
    html_dir = os.path.join(raw, "regulatory", "html")
    if os.path.exists(html_dir):
        for f in os.listdir(html_dir):
            if f.endswith(".html"):
                doc_ids.append(f[:-5])
    # 跳过 attachments (130个PDF, 58.8MB) - 太大导致OOM
    return doc_ids


# ============ 加载B榜题目 ============

def load_b_questions() -> List[dict]:
    """加载所有B榜题目"""
    questions = []
    for fname in sorted(os.listdir(B_QUESTION_DIR)):
        fpath = os.path.join(B_QUESTION_DIR, fname)
        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            questions.extend(data)
        elif fname.endswith(".jsonl"):
            with open(fpath, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        questions.append(json.loads(line))
    print(f"加载 {len(questions)} 道B榜题目")
    return questions


# ============ 全领域BM25检索 ============

class DomainRetriever:
    """领域级BM25检索器 - 缓存索引避免重复构建"""
    _instances: Dict[str, "DomainRetriever"] = {}

    def __init__(self, domain: str):
        self.domain = domain
        self.chunks: List[str] = []
        self.chunk_doc_ids: List[str] = []
        self.bm25 = None
        self._build()

    @classmethod
    def get(cls, domain: str) -> "DomainRetriever":
        if domain not in cls._instances:
            cls._instances[domain] = cls(domain)
        return cls._instances[domain]

    def _build(self):
        """构建领域全量BM25索引"""
        if self.domain == "regulatory":
            doc_ids = get_regulatory_doc_ids()
        else:
            doc_ids = DOMAIN_DOC_IDS.get(self.domain, [])

        print(f"  [{self.domain}] 构建索引: {len(doc_ids)} 个文档...")
        t0 = time.time()

        all_chunks = []
        all_doc_ids = []

        for doc_id in doc_ids:
            doc_chunks = build_doc_chunks(doc_id, self.domain)
            for chunk_text, idx in doc_chunks:
                all_chunks.append(chunk_text)
                all_doc_ids.append(doc_id)

        self.chunks = all_chunks
        self.chunk_doc_ids = all_doc_ids

        if all_chunks:
            tokenized = [tokenize(c) for c in all_chunks]
            self.bm25 = BM25Okapi(tokenized)

        t1 = time.time()
        print(f"  [{self.domain}] 索引完成: {len(all_chunks)} chunks, {t1-t0:.1f}s")

    def retrieve(self, query: str, top_k: int = None, max_chars: int = None) -> str:
        """检索并返回带标注的上下文"""
        if not self.bm25 or not self.chunks:
            return ""

        domain_cfg = config.DOMAIN_CONFIG.get(self.domain, {})
        if top_k is None:
            top_k = domain_cfg.get("top_k", config.TOP_K)
        if max_chars is None:
            max_chars = domain_cfg.get("max_ctx", config.MAX_CONTEXT_CHARS)

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 按分数排序
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])

        # 取 top_k 且不超过 max_chars，去重相邻块
        results = []
        total_chars = 0
        seen_docs = set()
        for idx, score in ranked[:top_k * 3]:
            if score <= 0:
                break
            chunk = self.chunks[idx]
            # 简单去重：同文档相邻块只保留一次
            doc_id = self.chunk_doc_ids[idx]
            chunk_key = chunk[:100]
            if chunk_key in seen_docs:
                continue
            seen_docs.add(chunk_key)

            if total_chars + len(chunk) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 200:
                    results.append(chunk[:remaining])
                break
            results.append(chunk)
            total_chars += len(chunk)
            if len(results) >= top_k:
                break

        # 组装带标注的上下文
        context_parts = []
        for i, chunk in enumerate(results):
            context_parts.append(f"[证据{i+1}]\n{chunk}")

        return "\n\n".join(context_parts)


# ============ Prompt设计 ============

SYSTEM_PROMPT_CHOICE = """你是金融文档审核专家，严格依据提供的证据材料作答。铁律：
1. 只依据证据内容判断，不用常识或外部知识替代证据。
2. 逐选项分析，每个选项必须在证据中找到明确支持或反驳依据。
3. 注意数字、期限、比例、条件等细节，警惕干扰项的细微篡改。
4. 多选题：选出所有正确选项，漏选错选均不得分。宁缺勿滥，对每个选项独立判断。
5. 判断题：A表示正确/是，B表示错误/否（以题目选项说明为准）。
6. 最终答案只输出大写字母，不加任何解释。
7. 若证据不足以判断某选项，倾向于不选（保守策略）。"""

SYSTEM_PROMPT_CALC = """你是金融文档计算专家，严格依据提供的证据材料进行计算。铁律：
1. 只依据证据中的数据和公式计算，不用外部知识。
2. 仔细查找题目所需的所有数据，确保数据来源正确。
3. 计算过程要精确，注意单位换算。
4. 严格按题目要求的格式输出最终答案。
5. 如果题目要求保留小数位，严格遵守。
6. 最终答案行格式：答案：XXX"""


def build_choice_prompt(question: str, options: dict, q_type: str, context: str) -> str:
    """构建选择题/判断题prompt"""
    opts_text = ""
    for key in sorted(options.keys()):
        opts_text += f"{key}. {options[key]}\n"

    if q_type == "单选题":
        fmt_hint = "单选题，选择唯一正确答案，输出一个大写字母（A/B/C/D）。"
    elif q_type == "多选题":
        fmt_hint = "多选题，选择所有正确答案，输出多个大写字母按字母序连续排列（如ACD），不用分隔符。"
    elif q_type == "判断题":
        fmt_hint = "判断题，A表示正确，B表示错误，输出一个大写字母（A或B）。"
    else:
        fmt_hint = "输出答案字母。"

    prompt = f"""## 证据材料
{context}

## 题目
{question}

## 选项
{opts_text}
## 要求
{fmt_hint}
请逐选项简要分析（每项≤15字，引用[证据N]），然后在最后一行输出答案。
格式：答案：XX"""
    return prompt


def build_calc_prompt(question: str, context: str) -> str:
    """构建计算题/抽取题prompt"""
    prompt = f"""## 证据材料
{context}

## 题目
{question}

## 作答要求
- 从证据中找到所需数据，简要列出关键数值和计算式（不超过5行）。
- 严格按题目要求的格式输出最终答案。
- 数字保留题目要求的小数位，不带单位（除非题目明确要求带%）。
- 日期格式：YYYY年M月D日。排序用英文>连接无空格。多答案用英文分号;分隔。
- 最后一行必须是：答案：XXX"""
    return prompt


# ============ 答案提取 ============

def extract_choice_answer(response: str, q_type: str) -> str:
    """从模型输出中提取选择题答案字母"""
    if not response:
        return ""

    # 策略1: 找"答案：XX"或"答案:XX"模式
    m = re.search(r"答案[：:]\s*([A-D]+)", response)
    if m:
        ans = m.group(1)
        return _normalize_choice(ans, q_type)

    # 策略2: 找最后一行中的连续大写字母
    lines = response.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r"\b([A-D]{1,4})\b", line)
        if m:
            ans = m.group(1)
            return _normalize_choice(ans, q_type)

    # 策略3: 全文找所有孤立大写字母
    letters = re.findall(r"(?<![a-zA-Z])([A-D])(?![a-zA-Z])", response)
    if letters:
        ans = "".join(letters)
        return _normalize_choice(ans, q_type)

    return ""


def _normalize_choice(ans: str, q_type: str) -> str:
    """规范化选择题答案"""
    ans = ans.upper()
    ans = re.sub(r"[^A-D]", "", ans)

    if q_type in ("单选题", "判断题"):
        return ans[0] if ans else ""
    elif q_type == "多选题":
        unique = sorted(set(ans))
        return "".join(unique)
    return ans


def extract_calc_answer(response: str, question: str) -> str:
    """从模型输出中提取计算题答案"""
    if not response:
        return ""

    # 策略1: 找"答案：XXX"或"答案:XXX"
    m = re.search(r"答案[：:]\s*(.+?)(?:\n|$)", response)
    if m:
        return m.group(1).strip()

    # 策略2: 找最后一行非空内容
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        # 去掉可能的前缀
        last = re.sub(r"^(最终答案|答案|结果)[：:]?\s*", "", last)
        return last.strip()

    return ""


def split_multi_answers(answer: str, question: str) -> List[str]:
    """将多答案题的答案拆分为列表"""
    if not answer:
        return [""]

    # 按分号分隔（中文分号或英文分号）
    parts = re.split(r"[;；]", answer)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) > 1:
        return parts

    # 如果没有分号，尝试按题目中的格式要求拆分
    # 有些答案用"；"分隔
    return [answer]


# ============ 单题作答 ============

def solve_question_b(q: dict) -> dict:
    """解答单道B榜题目"""
    qid = q["qid"]
    domain = q["domain"]
    question = q["question"]
    q_type = q["type"]
    options = q.get("options", {})

    print(f"\n{'='*60}")
    print(f"[{qid}] ({domain}/{q_type}) {question[:60]}...")

    # 1. 全领域检索
    t0 = time.time()
    retriever = DomainRetriever.get(domain)

    # 构建查询
    if options:
        query = build_query(question, options)
    else:
        query = question

    context = retriever.retrieve(query)
    t1 = time.time()
    print(f"  检索完成: {len(context)} chars, {t1-t0:.1f}s")

    if not context:
        print(f"  [WARN] 无有效上下文")
        return {
            "qid": qid,
            "answers": [""],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    # 2. 根据题型选择不同策略
    total_pt, total_ct = 0, 0

    if q_type in ("单选题", "多选题", "判断题"):
        # 选择题/判断题
        user_prompt = build_choice_prompt(question, options, q_type, context)

        if q_type == "多选题":
            # 多选题：3次调用+投票
            answers_list = []
            for i in range(3):
                temp = config.TEMPERATURE if i == 0 else 0.3
                response, pt, ct = call_llm_b(SYSTEM_PROMPT_CHOICE, user_prompt,
                                              max_tokens=MAX_TOKENS_CHOICE, temperature=temp)
                total_pt += pt
                total_ct += ct
                ans = extract_choice_answer(response, q_type)
                answers_list.append(ans)
                if i == 1 and answers_list[0] and answers_list[0] == answers_list[1]:
                    break
            answer = _vote_multi(answers_list)
        else:
            # 单选/判断：单次调用
            response, pt, ct = call_llm_b(SYSTEM_PROMPT_CHOICE, user_prompt,
                                          max_tokens=MAX_TOKENS_CHOICE)
            total_pt += pt
            total_ct += ct
            answer = extract_choice_answer(response, q_type)

        # fallback
        if not answer:
            fallback = f"基于证据直接输出答案字母。\n证据：{context[:3000]}\n题目：{question}\n选项：\n"
            fallback += "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
            fallback += "\n答案（只输出字母）："
            response2, pt2, ct2 = call_llm_b("你是金融专家，直接输出答案字母。", fallback,
                                             max_tokens=50)
            answer = extract_choice_answer(response2, q_type)
            total_pt += pt2
            total_ct += ct2

        answers = [answer] if answer else ["A"]

    else:
        # 计算题/抽取题
        user_prompt = build_calc_prompt(question, context)
        response, pt, ct = call_llm_b(SYSTEM_PROMPT_CALC, user_prompt,
                                      max_tokens=MAX_TOKENS_CALC)
        total_pt += pt
        total_ct += ct
        raw_answer = extract_calc_answer(response, question)
        print(f"  原始答案: {raw_answer}")

        # 拆分为多答案
        answers = split_multi_answers(raw_answer, question)

        # 如果提取失败，重试一次
        if not answers or not answers[0]:
            fallback_prompt = f"基于证据计算并直接输出答案。\n证据：{context[:5000]}\n题目：{question}\n答案："
            response2, pt2, ct2 = call_llm_b(SYSTEM_PROMPT_CALC, fallback_prompt,
                                             max_tokens=MAX_TOKENS_CALC)
            raw_answer2 = extract_calc_answer(response2, question)
            answers = split_multi_answers(raw_answer2, question)
            total_pt += pt2
            total_ct += ct2

    t2 = time.time()
    print(f"  答案: {answers} | tokens: {total_pt}+{total_ct}={total_pt+total_ct} | {t2-t1:.1f}s")

    return {
        "qid": qid,
        "answers": answers,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
    }


# ============ 生成CSV ============

def generate_csv_b(results: List[dict], output_path: str = None):
    """生成B榜提交CSV"""
    if output_path is None:
        output_path = B_OUTPUT_CSV

    total_pt = sum(r["prompt_tokens"] for r in results)
    total_ct = sum(r["completion_tokens"] for r in results)
    total_tokens = total_pt + total_ct

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer_1", "answer_2", "answer_3", "answer_4",
                         "prompt_tokens", "completion_tokens", "total_tokens"])
        # summary行
        writer.writerow(["summary", "", "", "", "", total_pt, total_ct, total_tokens])
        # 每题
        for r in results:
            answers = r["answers"]
            # 补齐4个答案位
            a1 = answers[0] if len(answers) > 0 else ""
            a2 = answers[1] if len(answers) > 1 else ""
            a3 = answers[2] if len(answers) > 2 else ""
            a4 = answers[3] if len(answers) > 3 else ""
            writer.writerow([r["qid"], a1, a2, a3, a4,
                           r["prompt_tokens"], r["completion_tokens"], r["total_tokens"]])

    print(f"\n{'='*60}")
    print(f"CSV 已生成: {output_path}")
    print(f"总 Token: {total_tokens:,} (prompt={total_pt:,}, completion={total_ct:,})")
    token_score = max(0, min(1, (5_000_000 - total_tokens) / 5_000_000))
    print(f"TokenScore: {token_score:.4f}")
    print(f"预估得分系数: {0.7 + 0.3 * token_score:.4f}")


# ============ 主流程 ============

def main():
    # 检查 API Key
    if not config.DASHSCOPE_API_KEY:
        print("[ERROR] 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    print("=" * 60)
    print("金融长文档 QA Agent - B榜")
    print(f"模型: {config.MODEL_NAME}")
    print(f"数据: {config.DATA_DIR}")
    print("=" * 60)

    # 1. 加载题目
    questions = load_b_questions()
    if not questions:
        print("[ERROR] 未找到B榜题目")
        sys.exit(1)

    # 2. 构建各领域BM25索引（内部按需加载文档并缓存）
    print("\n--- 构建BM25索引 ---")
    domains_needed = set(q["domain"] for q in questions)
    for domain in sorted(domains_needed):
        DomainRetriever.get(domain)

    # 4. 逐题作答
    print("\n--- 开始作答 ---")
    results = []
    for i, q in enumerate(questions):
        print(f"\n进度: {i+1}/{len(questions)}")
        result = solve_question_b(q)
        results.append(result)

    # 5. 生成CSV
    generate_csv_b(results)

    # 6. 打印汇总
    print("\n--- 作答汇总 ---")
    for r in results:
        print(f"  {r['qid']}: {r['answers']} ({r['total_tokens']} tokens)")


if __name__ == "__main__":
    main()
