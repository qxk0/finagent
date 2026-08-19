"""BM25 检索模块 - 金融词典增强 + jieba 分词"""
import re
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

import config
from preprocess import build_doc_chunks

# ============ 金融专业词典 ============
FINANCE_TERMS = [
    # 保险
    "身故保险金", "现金价值", "保单账户", "基本保额", "养老年金", "退保金额",
    "保险责任", "责任免除", "等待期", "犹豫期", "保费豁免", "生存保险金",
    "万能账户", "结算利率", "保障期间", "交费期间", "保险期间", "投保年龄",
    "受益人", "投保人", "被保险人", "保险金申请", "理赔", "免赔额",
    "给付比例", "丧葬补助金", "抚恤金", "一次性工亡补助金",
    # 监管法规
    "股东大会", "特别决议", "普通决议", "独立董事", "尽职调查", "受益所有人",
    "反洗钱", "可疑交易", "大额交易", "客户身份识别", "行政处罚", "监管措施",
    "信息披露", "内部控制", "合规管理", "风险提示", "备案登记",
    "资产负债率", "净资产", "募集资金", "担保对象", "关联交易",
    "表决权", "出席股东", "三分之二", "过半数", "章程指引", "治理准则",
    "施行日期", "废止", "过渡期", "整改期限",
    # 金融合同
    "募集说明书", "债券条款", "票面利率", "到期日", "付息日", "兑付",
    "信用评级", "担保方式", "抵押物", "质押物", "违约事件", "交叉违约",
    "加速到期", "回售权", "赎回权", "调整票面利率", "偿债保障措施",
    "受托管理人", "持有人会议", "发行规模", "期限品种",
    # 财务报表
    "营业收入", "净利润", "归属母公司", "经营活动现金流", "投资活动现金流",
    "筹资活动现金流", "研发投入", "资本化", "费用化", "毛利率", "净利率",
    "资产负债率", "流动比率", "速动比率", "应收账款", "存货周转",
    "商誉减值", "资产减值", "公允价值变动", "投资收益", "营业外收支",
    "分红方案", "每股股利", "派息", "送股", "转增",
    # 研报
    "盈利预测", "目标价", "投资评级", "买入评级", "增持评级",
    "行业景气度", "渗透率", "市占率", "同比增速", "环比增速",
    "市盈率", "市净率", "估值", "催化剂", "风险提示",
    # 通用金融
    "年化收益率", "复利", "单利", "折现率", "现值", "终值",
    "杠杆率", "资本充足率", "不良率", "拨备覆盖率", "流动性覆盖率",
]

# 加载词典
for term in FINANCE_TERMS:
    jieba.add_word(term, freq=100000)


def tokenize(text: str) -> List[str]:
    """金融增强分词"""
    # 先提取数字+单位模式（如 70%、100万元、30个工作日）
    text_clean = re.sub(r"\s+", " ", text)
    tokens = jieba.lcut(text_clean)
    # 过滤停用词和单字符
    stop = set("的了是在有和与及或者不也都而但 yet 这那其此该等对被把从到为以于中上下内外部全部")
    result = []
    for t in tokens:
        t = t.strip()
        if len(t) >= 2 or t.isdigit():
            if t not in stop:
                result.append(t)
    return result


class BM25Retriever:
    """针对单题的 BM25 检索器"""

    def __init__(self, doc_ids: List[str], domain: str):
        self.doc_ids = doc_ids
        self.domain = domain
        self.chunks: List[str] = []
        self.chunk_meta: List[Tuple[str, int]] = []  # (doc_id, chunk_idx)
        self.bm25 = None
        self._build_index()

    def _build_index(self):
        """构建 BM25 索引"""
        all_chunks = []
        all_meta = []

        for doc_id in self.doc_ids:
            doc_chunks = build_doc_chunks(doc_id, self.domain)
            for chunk_text, idx in doc_chunks:
                all_chunks.append(chunk_text)
                all_meta.append((doc_id, idx))

        self.chunks = all_chunks
        self.chunk_meta = all_meta

        if all_chunks:
            tokenized = [tokenize(c) for c in all_chunks]
            self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = None, max_chars: int = None) -> List[str]:
        """检索最相关的文本块"""
        if not self.bm25 or not self.chunks:
            return []

        if top_k is None:
            top_k = config.TOP_K
        if max_chars is None:
            max_chars = config.MAX_CONTEXT_CHARS

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # 按分数排序
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])

        # 取 top_k 且不超过 max_chars
        results = []
        total_chars = 0
        for idx, score in ranked[:top_k * 2]:  # 多取一些备选
            if score <= 0:
                break
            chunk = self.chunks[idx]
            if total_chars + len(chunk) > max_chars:
                # 尝试截断最后一块
                remaining = max_chars - total_chars
                if remaining > 200:
                    results.append(chunk[:remaining])
                break
            results.append(chunk)
            total_chars += len(chunk)
            if len(results) >= top_k:
                break

        return results

    def retrieve_for_question(self, question: str, options: dict,
                              top_k: int = None, max_chars: int = None) -> str:
        """为题目构建检索上下文"""
        # 构建查询：题干 + 所有选项
        query_parts = [question]
        for key in sorted(options.keys()):
            query_parts.append(options[key])
        query = " ".join(query_parts)

        chunks = self.retrieve(query, top_k=top_k, max_chars=max_chars)

        # 组装上下文，标注来源
        context_parts = []
        for i, chunk in enumerate(chunks):
            doc_id, chunk_idx = self.chunk_meta[
                sorted(range(len(self.chunks)),
                       key=lambda x: -self.bm25.get_scores(tokenize(query))[x])[i]
            ] if i < len(self.chunk_meta) else ("unknown", 0)
            context_parts.append(f"[证据{i+1}] {chunk}")

        return "\n\n".join(context_parts)


def build_query(question: str, options: dict) -> str:
    """构建检索查询字符串"""
    parts = [question]
    for key in sorted(options.keys()):
        parts.append(f"{key}. {options[key]}")
    return "\n".join(parts)


def retrieve_context(doc_ids: List[str], domain: str, question: str,
                     options: dict, top_k: int = None, max_chars: int = None) -> str:
    """一站式检索：给定文档列表和题目，返回拼接好的证据上下文"""
    domain_cfg = config.DOMAIN_CONFIG.get(domain, {})
    if top_k is None:
        top_k = domain_cfg.get("top_k", config.TOP_K)
    if max_chars is None:
        max_chars = domain_cfg.get("max_ctx", config.MAX_CONTEXT_CHARS)

    retriever = BM25Retriever(doc_ids, domain)

    # 构建查询
    query = build_query(question, options)
    chunks = retriever.retrieve(query, top_k=top_k, max_chars=max_chars)

    # 组装带标注的上下文
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[证据{i+1}]\n{chunk}")

    return "\n\n".join(context_parts)
