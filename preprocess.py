"""文档预处理模块 - PDF/HTML/TXT 解析与分块"""
import os
import re
import json
import hashlib
from typing import Dict, List, Tuple

import pdfplumber
from bs4 import BeautifulSoup

import config


def ensure_cache_dir():
    os.makedirs(config.CACHE_DIR, exist_ok=True)


def _cache_path(doc_id: str, domain: str) -> str:
    safe_id = hashlib.md5(doc_id.encode()).hexdigest()[:12]
    return os.path.join(config.CACHE_DIR, f"{domain}_{safe_id}.txt")


def _is_cached(doc_id: str, domain: str) -> bool:
    return os.path.exists(_cache_path(doc_id, domain))


def _read_cache(doc_id: str, domain: str) -> str:
    with open(_cache_path(doc_id, domain), "r", encoding="utf-8") as f:
        return f.read()


def _write_cache(doc_id: str, domain: str, text: str):
    with open(_cache_path(doc_id, domain), "w", encoding="utf-8") as f:
        f.write(text)


# ============ PDF 解析 ============

def extract_pdf(pdf_path: str) -> str:
    """用 pdfplumber 提取 PDF 全文，保留段落结构"""
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                # 尝试提取表格
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            cells = [str(c).strip() if c else "" for c in row]
                            row_text = " | ".join(cells)
                            if row_text.strip(" |"):
                                pages_text.append(row_text)
    except Exception as e:
        print(f"  [WARN] PDF解析失败 {pdf_path}: {e}")
    return "\n".join(pages_text)


# ============ HTML 解析 ============

def extract_html(html_path: str) -> str:
    """从 HTML 提取正文文本"""
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        # 移除脚本和样式
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # 清理多余空行
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"  [WARN] HTML解析失败 {html_path}: {e}")
        return ""


# ============ TXT 解析 ============

def extract_txt(txt_path: str) -> str:
    """读取纯文本文件"""
    try:
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except Exception:
        try:
            with open(txt_path, "r", encoding="gbk") as f:
                return f.read()
        except Exception as e:
            print(f"  [WARN] TXT读取失败 {txt_path}: {e}")
            return ""


# ============ 文档定位 ============

def resolve_doc_path(doc_id: str, domain: str) -> str:
    """根据 doc_id 和 domain 找到实际文件路径"""
    raw = config.RAW_DIR

    if domain == "insurance":
        return os.path.join(raw, "insurance", f"{doc_id}.pdf")

    elif domain == "financial_contracts":
        return os.path.join(raw, "financial_contracts", f"{doc_id}.pdf")

    elif domain == "financial_reports":
        # doc_id 如 annual_byd_2024_report，文件可能是 .PDF 或 .pdf
        base = os.path.join(raw, "financial_reports")
        for ext in [".PDF", ".pdf"]:
            p = os.path.join(base, doc_id + ext)
            if os.path.exists(p):
                return p
        return os.path.join(base, doc_id + ".PDF")

    elif domain == "research":
        return os.path.join(raw, "research", f"{doc_id}.pdf")

    elif domain == "regulatory":
        # 三种情况: strict_v3_xxx (txt), csrc_XXXX (html), csrc_XXXX_attN (pdf)
        if doc_id.startswith("strict_v3_"):
            txt_dir = os.path.join(raw, "regulatory", "txt")
            return os.path.join(txt_dir, doc_id + ".txt")
        elif "_att" in doc_id:
            # csrc_0009_att1 -> attachments/csrc_0009_att1.pdf
            att_dir = os.path.join(raw, "regulatory", "attachments")
            return os.path.join(att_dir, doc_id + ".pdf")
        else:
            # csrc_0262 -> html/csrc_0262.html
            html_dir = os.path.join(raw, "regulatory", "html")
            return os.path.join(html_dir, doc_id + ".html")

    return ""


def load_document(doc_id: str, domain: str) -> str:
    """加载文档文本（带缓存）"""
    ensure_cache_dir()
    if _is_cached(doc_id, domain):
        return _read_cache(doc_id, domain)

    path = resolve_doc_path(doc_id, domain)
    if not os.path.exists(path):
        print(f"  [WARN] 文件不存在: {path}")
        return ""

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_pdf(path)
    elif ext == ".html":
        text = extract_html(path)
    elif ext == ".txt":
        text = extract_txt(path)
    else:
        text = ""

    _write_cache(doc_id, domain, text)
    return text


# ============ 分块 ============

def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """将文本按段落优先、长度限制进行分块"""
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    if overlap is None:
        overlap = config.CHUNK_OVERLAP

    if not text.strip():
        return []

    # 先按段落分割
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current = ""

    for para in paragraphs:
        # 如果单个段落就超过 chunk_size，强制切分
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            # 按句子切分长段落
            sentences = re.split(r"(?<=[。；;！!？?\n])", para)
            sub = ""
            for sent in sentences:
                if len(sub) + len(sent) > chunk_size:
                    if sub:
                        chunks.append(sub)
                    sub = sent
                else:
                    sub += sent
            if sub:
                current = sub
        elif len(current) + len(para) + 1 > chunk_size:
            chunks.append(current)
            # 重叠：保留尾部
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n" + para
            else:
                current = para
        else:
            current = current + "\n" + para if current else para

    if current:
        chunks.append(current)

    return chunks


def build_doc_chunks(doc_id: str, domain: str) -> List[Tuple[str, int]]:
    """加载文档并分块，返回 [(chunk_text, chunk_idx), ...]"""
    text = load_document(doc_id, domain)
    if not text:
        return []
    chunks = chunk_text(text)
    return [(c, i) for i, c in enumerate(chunks)]


# ============ 批量预处理 ============

def preprocess_all(questions: List[dict]):
    """预加载所有题目涉及的文档到缓存"""
    seen = set()
    for q in questions:
        domain = q["domain"]
        for doc_id in q.get("doc_ids", []):
            key = (doc_id, domain)
            if key not in seen:
                seen.add(key)
                print(f"  预处理: [{domain}] {doc_id}")
                load_document(doc_id, domain)
    print(f"  共预处理 {len(seen)} 个文档")


if __name__ == "__main__":
    # 测试
    ensure_cache_dir()
    text = load_document("strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）", "regulatory")
    print(f"文本长度: {len(text)}")
    chunks = chunk_text(text)
    print(f"分块数: {len(chunks)}")
    print(f"第一块: {chunks[0][:200]}")
