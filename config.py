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
