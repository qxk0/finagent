"""推理作答模块 - Prompt设计 + API调用 + 答案提取"""
import re
import time
from typing import Tuple

from openai import OpenAI

import config

client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url=config.BASE_URL,
)

# ============ System Prompt ============
SYSTEM_PROMPT = """你是金融文档审核专家，严格依据提供的证据材料作答。铁律：
1. 只依据证据内容判断，不用常识或外部知识替代证据。
2. 逐选项分析，每个选项必须在证据中找到明确支持或反驳依据。
3. 注意数字、期限、比例、条件等细节，警惕干扰项的细微篡改。
4. 多选题：选出所有正确选项，漏选错选均不得分。
5. 判断题：A表示正确/是，B表示错误/否（以题目选项说明为准）。
6. 最终答案只输出大写字母，不加任何解释。
7. 若证据不足以判断某选项，倾向于不选（保守策略）。"""


def build_user_prompt(question: str, options: dict, answer_format: str,
                      context: str) -> str:
    """构建用户 prompt"""
    # 格式化选项
    opts_text = ""
    for key in sorted(options.keys()):
        opts_text += f"{key}. {options[key]}\n"

    # 题型说明
    if answer_format == "mcq":
        fmt_hint = "单选题，选择唯一正确答案，输出一个大写字母（A/B/C/D）。"
    elif answer_format == "multi":
        fmt_hint = "多选题，选择所有正确答案，输出多个大写字母按字母序排列（如ABC）。"
    elif answer_format == "tf":
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
请先逐选项简要分析（每项≤15字），然后在最后一行输出答案。
格式：答案：XX"""

    return prompt


def call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, int, int]:
    """调用 LLM API，返回 (response_text, prompt_tokens, completion_tokens)"""
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
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


def extract_answer(response: str, answer_format: str) -> str:
    """从模型输出中提取答案字母"""
    if not response:
        return ""

    # 策略1: 找"答案：XX"或"答案:XX"模式
    m = re.search(r"答案[：:]\s*([A-D]+)", response)
    if m:
        ans = m.group(1)
        return _normalize_answer(ans, answer_format)

    # 策略2: 找最后一行中的连续大写字母
    lines = response.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r"\b([A-D]{1,4})\b", line)
        if m:
            ans = m.group(1)
            return _normalize_answer(ans, answer_format)

    # 策略3: 全文找所有孤立大写字母
    letters = re.findall(r"(?<![a-zA-Z])([A-D])(?![a-zA-Z])", response)
    if letters:
        ans = "".join(letters)
        return _normalize_answer(ans, answer_format)

    return ""


def _normalize_answer(ans: str, answer_format: str) -> str:
    """规范化答案"""
    # 去重、排序、只保留A-D
    ans = ans.upper()
    ans = re.sub(r"[^A-D]", "", ans)

    if answer_format == "mcq" or answer_format == "tf":
        # 单选/判断：只取第一个
        return ans[0] if ans else ""
    elif answer_format == "multi":
        # 多选：去重排序
        unique = sorted(set(ans))
        return "".join(unique)
    return ans


def _vote_multi(answers: list) -> str:
    """对多选题的多次结果进行投票，取出现次数最多的答案"""
    from collections import Counter
    if not answers:
        return ""
    valid = [a for a in answers if a]
    if not valid:
        return ""
    counter = Counter(valid)
    # 取出现次数最多的；若平票，取字母数较多的（偏保守全选）
    best = counter.most_common()
    max_count = best[0][1]
    tied = [ans for ans, cnt in best if cnt == max_count]
    if len(tied) == 1:
        return tied[0]
    # 平票时取字母数最少的（宁缺勿滥）：赛题漏选/错选/多选均计为错误，
    # 过度选择（多选）会导致整题失分，保守策略与 system prompt 一致。
    return sorted(tied, key=lambda x: len(x))[0]


def answer_question(question: str, options: dict, answer_format: str,
                    context: str) -> Tuple[str, int, int]:
    """完整的单题作答流程，返回 (answer, prompt_tokens, completion_tokens)
    
    多选题使用3次调用+投票机制提升准确率。
    单选题/判断题使用单次调用节省token。
    """
    user_prompt = build_user_prompt(question, options, answer_format, context)
    total_pt, total_ct = 0, 0

    if answer_format == "multi":
        # 多选题：3次调用 + 投票
        answers = []
        for i in range(3):
            # 第2、3次用略高温度增加多样性
            temp = config.TEMPERATURE if i == 0 else 0.3
            response, pt, ct = _call_llm_with_temp(SYSTEM_PROMPT, user_prompt, temp)
            total_pt += pt
            total_ct += ct
            ans = extract_answer(response, answer_format)
            answers.append(ans)
            # 如果前两次一致，提前结束节省token
            if i == 1 and answers[0] and answers[0] == answers[1]:
                return answers[0], total_pt, total_ct

        answer = _vote_multi(answers)
    else:
        # 单选/判断：单次调用
        response, pt, ct = call_llm(SYSTEM_PROMPT, user_prompt)
        total_pt += pt
        total_ct += ct
        answer = extract_answer(response, answer_format)

    # 如果提取失败，尝试二次调用（简化prompt）
    if not answer:
        fallback_prompt = f"""基于以下证据回答问题，直接输出答案字母。

证据摘要：{context[:3000]}

题目：{question}
选项：
{chr(10).join(f'{k}. {v}' for k, v in sorted(options.items()))}

答案（只输出字母）："""
        response2, pt2, ct2 = call_llm("你是金融专家，直接输出答案字母。", fallback_prompt)
        answer = extract_answer(response2, answer_format)
        total_pt += pt2
        total_ct += ct2

    return answer, total_pt, total_ct


def _call_llm_with_temp(system_prompt: str, user_prompt: str,
                        temperature: float) -> Tuple[str, int, int]:
    """带自定义温度的 LLM 调用"""
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=config.MAX_TOKENS,
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
