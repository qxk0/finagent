"""配置文件 - 金融长文档QA Agent"""
import os

# === API 配置 ===
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
MODEL_NAME = "qwen-plus"  # 基准模型
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# === 路径配置 ===
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_dataset_upload")
QUESTIONS_DIR = os.path.join(DATA_DIR, "questions", "group_a")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer.csv")

# === 检索配置 ===
CHUNK_SIZE = 600          # 每块字符数
CHUNK_OVERLAP = 100       # 块重叠
TOP_K = 20                # BM25 检索返回块数
MAX_CONTEXT_CHARS = 55000  # 最大上下文字符数（约27K tokens）

# === 领域特殊配置 ===
DOMAIN_CONFIG = {
    "insurance": {"top_k": 20, "max_ctx": 45000},
    "regulatory": {"top_k": 25, "max_ctx": 50000},
    "financial_contracts": {"top_k": 20, "max_ctx": 50000},
    "financial_reports": {"top_k": 30, "max_ctx": 65000},
    "research": {"top_k": 25, "max_ctx": 55000},
}

# === 推理配置 ===
TEMPERATURE = 0.1
MAX_TOKENS = 300          # 输出token上限（答案很短）
TIMEOUT = 120             # API超时秒数
MAX_RETRIES = 3           # 重试次数

# === 动态记忆压缩 / 动态上下文预算（任务1、2）===
# 是否默认启用动态预算 + 记忆压缩（不影响 answer.csv 默认输出，由新入口显式传入）
USE_DYNAMIC_BUDGET = True
USE_MEMORY_COMPRESSION = True

# 三档预算参数。**这些值由 experiments/tune_budget.py 在真实数据上实测选择后
# 写入 experiments/budget_tiers.json**，此处的字典仅作为文件缺失时的兜底默认值。
BUDGET_TIERS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiments", "budget_tiers.json"
)
# B 榜（全领域检索、计算题为主）单独调参的结果；由 tune_budget.py --split B 生成
BUDGET_TIERS_FILE_B = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiments", "budget_tuning_b.json"
)
BUDGET_TIERS = {
    "simple":  {"top_k": 10, "max_chars": 12000, "compression_ratio": 0.60},
    "normal":  {"top_k": 20, "max_chars": 20000, "compression_ratio": 0.50},
    "complex": {"top_k": 30, "max_chars": 32000, "compression_ratio": 0.40},
}

# === 实验与日志（任务3、4、5、6、7）===
EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
LOG_DIR = os.path.join(EXPERIMENTS_DIR, "logs")
SELF_CONSISTENCY_LOG = os.path.join(LOG_DIR, "self_consistency.jsonl")
RETRIEVAL_LOG = os.path.join(LOG_DIR, "retrieval_budget.jsonl")
RETRY_LOG = os.path.join(LOG_DIR, "failure_retrieval.jsonl")
RESULTS_CSV = os.path.join(EXPERIMENTS_DIR, "results.csv")

# 自一致性配置（与 reasoner 现有多选逻辑一致：3 次采样，2 票一致提前停止）
SC_N_SAMPLES = 3
SC_TEMPERATURES = (0.1, 0.3, 0.3)

# === Full-text baseline 配置（任务4）===
# qwen-plus 上下文窗口按 128K 计，留出输出与模板余量
FULLTEXT_MAX_INPUT_TOKENS = 120000

# === 失败驱动重检索（任务6）===
MAX_RETRY_PER_QUESTION = 1
RETRY_TOP_K_MULTIPLIER = 2
