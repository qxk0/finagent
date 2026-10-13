"""动态记忆压缩问答（Dynamic Memory Compression QA）子包。

模块划分：
  text_features.py     题目/文本的确定性特征与"关键跨度"（数字、金额、日期、条款号、实体）
  context_budget.py    根据题目特征动态决定 top_k / max_chars（simple/normal/complex）
  memory_compressor.py 确定性证据压缩（句子级保留，绝不粗暴截断），可追溯到原 chunk
  token_estimator.py   基于真实 API 调用标定的 prompt token 估计器（离线实验用）
  self_consistency.py  多选题多次采样 + 提前停止 + 结构化日志
  failure_retrieval.py 首轮作答 → 失败检测 → 定向重检索 → 二次作答
  pipeline.py          串起上述模块的问答流水线（BM25 → 证据打分过滤 → 记忆压缩 → 上下文）
"""

__all__ = [
    "text_features",
    "context_budget",
    "memory_compressor",
    "token_estimator",
    "self_consistency",
    "failure_retrieval",
    "pipeline",
]
