"""领域语料清单（从 main_b.py 抽出来，便于在无 API Key 环境下使用）。

main_b.py 在 import 时会创建 OpenAI client，因此无 Key 环境下无法被其他模块复用；
这里把"领域 → doc_id 列表"的纯数据逻辑抽出来，行为与 main_b.py 完全一致。
"""
import os
from typing import Dict, List

import config

DOMAIN_DOC_IDS: Dict[str, List[str]] = {
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


def get_regulatory_doc_ids(include_attachments: bool = False) -> List[str]:
    """regulatory 领域 doc_id（txt + html；默认跳过大型 PDF 附件，避免 OOM）。"""
    raw = config.RAW_DIR
    doc_ids: List[str] = []
    txt_dir = os.path.join(raw, "regulatory", "txt")
    if os.path.exists(txt_dir):
        doc_ids += [f[:-4] for f in os.listdir(txt_dir) if f.endswith(".txt")]
    html_dir = os.path.join(raw, "regulatory", "html")
    if os.path.exists(html_dir):
        doc_ids += [f[:-5] for f in os.listdir(html_dir) if f.endswith(".html")]
    if include_attachments:
        att_dir = os.path.join(raw, "regulatory", "attachments")
        if os.path.exists(att_dir):
            doc_ids += [f[:-4] for f in os.listdir(att_dir) if f.lower().endswith(".pdf")]
    return doc_ids


def domain_doc_ids(domain: str) -> List[str]:
    if domain == "regulatory":
        return get_regulatory_doc_ids()
    return list(DOMAIN_DOC_IDS.get(domain, []))
