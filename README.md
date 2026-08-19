# 金融长文档智能问答系统

基于 RAG（检索增强生成）的金融长文档问答系统，AFAC2026 金融智能创新大赛·赛题四
（金融长文本 Agent 的动态记忆压缩与高效问答挑战）参赛项目，覆盖 5 大金融领域：
保险条款、监管法规、金融合同、上市公司财报、行业研报。

## 系统流程

```
题目加载 → 文档解析(PDF/HTML/TXT) → 分块 → BM25 检索证据
        → Qwen 大模型证据推理 → 答案规范化 → answer.csv
```

## 目录结构

```
main.py          A 榜主流程（题目自带 doc_ids）
main_b.py        B 榜主流程（全领域检索，含计算题/抽取题）
preprocess.py    文档解析与分块（pdfplumber / BeautifulSoup + 缓存）
retriever.py     BM25 检索 + jieba 金融词典增强
reasoner.py      Prompt 设计、Qwen API 调用、答案提取与投票
config.py        配置（API Key 从环境变量读取）
evaluate.py      自建评估集脚本（需 eval_labels.json 提供标准答案）
```

## 快速开始

```bash
pip install -r requirements.txt

# 设置 API Key（阿里云百炼 DashScope）
export DASHSCOPE_API_KEY="sk-xxx"        # Windows: set DASHSCOPE_API_KEY=sk-xxx

# A 榜：题目自带关联文档
python main.py

# B 榜：全领域检索
python main_b.py
```

运行后生成 `answer.csv` / `answer_b.csv`，包含每题答案与 Token 消耗统计。

## 主要设计

- **检索**：jieba 分词 + BM25，内置 133 个金融术语词典（保单账户价值、资产减值等）；
  按领域分别调优 top-k 与上下文长度。
- **分块**：段落优先、定长截断、100 字符重叠，兼顾语义完整性与检索粒度。
- **推理**：证据引用 + 逐项分析 + 格式约束的 prompt；单选/多选/判断分题型处理；
  多选题 3 次采样投票，平票时保守选择（宁缺勿滥）。
- **成本控制**：文档缓存、分题型输出长度限制、投票一致提前终止；
  A 组 100 题总 Token 167 万、B 组 200 万（预算 500 万）。

## 结果

| 组别 | 题数 | 总 Token | 说明 |
|---|---|---|---|
| A 榜 | 100 | 1,673,137 | 官方基线 Token 约 299 万 |
| B 榜 | 100 | 2,002,397 | 全领域盲测 |

## 数据说明

比赛数据（`public_dataset_upload/`、`upload_b/`）按比赛保密要求**不随仓库发布**。
仓库仅包含代码与示例输出。
