"""动态记忆压缩问答流水线（任务1 + 任务2 的落地实现）。

    Question
      → BM25 candidate chunks                (候选召回，top_k 由动态预算决定)
      → Evidence scoring / filtering          (BM25 + 关键跨度覆盖 + 词覆盖 → 过滤/去重)
      → Dynamic context budget                (simple/normal/complex → top_k / max_chars)
      → Memory compression                    (句子级压缩，保护数字/金额/日期/条款/实体)
      → Compressed context (可追溯)           (每条证据带 doc_id/chunk_id/page)
      → Qwen reasoning                        (沿用原有 reasoner)

设计上刻意保持"可解释、可复现"：不引入 embedding、不引入额外 LLM 摘要、
不做字符串截断；超预算时丢弃整块，而不是切半句话。
"""
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

import config
from preprocess import build_doc_chunks, build_paged_chunks, load_document, load_document_pages
from retriever import FINANCE_TERMS, build_query, tokenize

from .context_budget import ContextBudget, decide_budget, load_tier_params
from .corpus import domain_doc_ids
from .memory_compressor import (CompressedEvidence, CompressionResult, EvidenceChunk,
                                build_context, compress_evidence, filter_evidence,
                                score_evidence_chunks, verify_traceability)
from .text_features import (QuestionFeatures, extract_question_features, idf_weights,
                            span_coverage)
from .token_estimator import TokenEstimator, load_estimator


@dataclass
class SelectedChunkInfo:
    doc_id: str
    chunk_id: int
    page: Optional[int]
    chars_before: int
    chars_after: int
    bm25_score: float
    evidence_score: float
    mode: str                 # full | compressed
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "bm25_score": round(self.bm25_score, 4),
            "evidence_score": round(self.evidence_score, 4),
            "mode": self.mode,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class PipelineResult:
    qid: str
    domain: str
    budget: ContextBudget
    features: Dict[str, object]
    n_candidates: int
    selected: List[SelectedChunkInfo]
    context: str
    context_chars: int
    chars_before_compression: int
    compression_ratio: float
    key_span_hit_rate: float
    key_span_missing: List[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_source: str = "estimated"       # estimated | api
    use_compression: bool = True
    compression_stats: Dict[str, object] = field(default_factory=dict)
    traceability_problems: List[str] = field(default_factory=list)
    retrieval_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_log_record(self) -> Dict[str, object]:
        return {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qid": self.qid,
            "domain": self.domain,
            "tier": self.budget.tier,
            "top_k": self.budget.top_k,
            "max_chars": self.budget.max_chars,
            "use_compression": self.use_compression,
            "compression_target_ratio": self.budget.compression_ratio,
            "budget_reasons": list(self.budget.reasons),
            "budget_source": self.budget.source,
            "features": self.features,
            "n_candidates": self.n_candidates,
            "selected_chunks": [s.to_dict() for s in self.selected],
            "n_selected": len(self.selected),
            "context_chars": self.context_chars,
            "chars_before_compression": self.chars_before_compression,
            "compression_ratio": round(self.compression_ratio, 4),
            "compression_stats": self.compression_stats,
            "key_span_hit_rate": round(self.key_span_hit_rate, 4),
            "key_span_missing": list(self.key_span_missing),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "token_source": self.token_source,
            "retrieval_seconds": round(self.retrieval_seconds, 3),
            "traceability_problems": list(self.traceability_problems),
        }


def append_jsonl(path: str, record: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [WARN] 写日志失败 {path}: {e}")


class DynamicMemoryPipeline:
    """一道题的动态记忆压缩流水线（索引在实例内构建并复用）。"""

    _domain_index_cache: Dict[Tuple[str, bool, bool], Tuple[List[str], List[Tuple[str, int, Optional[int]]], object]] = {}
    # 文档级缓存：(domain, doc_id, page_aware, use_finance_dict) -> (chunks, meta, tokenized)
    # A 榜 100 题会反复用到同样的 68 篇文档，缓存掉 jieba 分词是最主要的提速手段。
    _doc_cache: Dict[Tuple[str, str, bool, bool], Tuple[List[str], List[Tuple[str, int, Optional[int]]], List[List[str]]]] = {}

    def __init__(self, domain: str, doc_ids: Optional[Sequence[str]] = None,
                 use_finance_dict: bool = True, page_aware: bool = False,
                 share_domain_index: bool = False,
                 tier_params: Optional[Dict[str, Dict[str, float]]] = None):
        self.domain = domain
        self.use_finance_dict = use_finance_dict
        self.page_aware = page_aware
        self.share_domain_index = share_domain_index
        self.tier_params = tier_params
        self.doc_ids: List[str] = list(doc_ids) if doc_ids else domain_doc_ids(domain)
        self.chunks: List[str] = []
        self.meta: List[Tuple[str, int, Optional[int]]] = []   # (doc_id, chunk_id, page)
        self.bm25 = None
        self._build_index()

    # ------------------------------------------------------------ 索引
    def _build_index(self):
        cache_key = (self.domain, self.use_finance_dict, self.page_aware)
        if self.share_domain_index and cache_key in self._domain_index_cache and not self._doc_ids_partial():
            self.chunks, self.meta, self.bm25 = self._domain_index_cache[cache_key]
            return

        chunks: List[str] = []
        meta: List[Tuple[str, int, Optional[int]]] = []
        tokenized: List[List[str]] = []
        for doc_id in self.doc_ids:
            d_chunks, d_meta, d_tokens = self._doc_chunks(doc_id)
            chunks.extend(d_chunks)
            meta.extend(d_meta)
            tokenized.extend(d_tokens)

        self.chunks = chunks
        self.meta = meta
        if chunks:
            self.bm25 = BM25Okapi(tokenized)
        if self.share_domain_index:
            self._domain_index_cache[cache_key] = (self.chunks, self.meta, self.bm25)

    def _doc_chunks(self, doc_id: str):
        """取单篇文档的 (chunk文本, (doc_id, chunk_id, page), 分词结果)，带进程内缓存。"""
        key = (self.domain, doc_id, self.page_aware, self.use_finance_dict)
        cached = self._doc_cache.get(key)
        if cached is not None:
            return cached
        chunks: List[str] = []
        meta: List[Tuple[str, int, Optional[int]]] = []
        if self.page_aware:
            for text, idx, page in build_paged_chunks(doc_id, self.domain):
                chunks.append(text)
                meta.append((doc_id, idx, page))
        else:
            for text, idx in build_doc_chunks(doc_id, self.domain):
                chunks.append(text)
                meta.append((doc_id, idx, None))
        tokens = [tokenize(c, use_finance_dict=self.use_finance_dict) for c in chunks]
        self._doc_cache[key] = (chunks, meta, tokens)
        return self._doc_cache[key]

    def _doc_ids_partial(self) -> bool:
        """当前 doc_ids 是否是领域全量语料的子集（子集索引不能复用全量索引）。"""
        full = domain_doc_ids(self.domain)
        return len(self.doc_ids) < len(full)

    # ------------------------------------------------------------ 检索
    def bm25_scores(self, query: str) -> List[float]:
        if not self.bm25:
            return []
        return list(self.bm25.get_scores(tokenize(query, use_finance_dict=self.use_finance_dict)))

    def candidate_chunks(self, query: str, n_candidates: int,
                         max_chars: Optional[int] = None) -> List[EvidenceChunk]:
        """按 BM25 取候选块（不做任何字符截断，超预算就整块丢弃）。"""
        scores = self.bm25_scores(query)
        if not scores:
            return []
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])
        out: List[EvidenceChunk] = []
        used = 0
        for idx, score in ranked:
            if score <= 0:
                break
            text = self.chunks[idx]
            if max_chars and used + len(text) > max_chars and out:
                continue          # 整块跳过，绝不切半个块
            doc_id, chunk_id, page = self.meta[idx]
            out.append(EvidenceChunk(doc_id=doc_id, chunk_id=chunk_id, text=text,
                                     page=page, bm25_score=float(score), rank=len(out)))
            used += len(text)
            if len(out) >= n_candidates:
                break
        return out

    # ------------------------------------------------------------ 主流程
    def run(self, question: str, options: Optional[dict] = None, *,
            qid: str = "", answer_format: Optional[str] = None,
            q_type: Optional[str] = None, budget: Optional[ContextBudget] = None,
            use_compression: Optional[bool] = None,
            query_extra: str = "",
            system_prompt: str = "", user_prompt: str = "",
            estimator: Optional[TokenEstimator] = None) -> PipelineResult:
        t0 = time.time()
        options = options or {}

        features: QuestionFeatures = extract_question_features(
            question, options, qid=qid, domain=self.domain, doc_ids=self.doc_ids,
            answer_format=answer_format, q_type=q_type)
        budget = budget or decide_budget(features, tier_params=self.tier_params,
                                        use_compression=use_compression)
        use_compression = budget.use_compression if use_compression is None else use_compression

        query = build_query(question, options) if options else question
        if query_extra:
            # 定向重检索用：把"题目里出现、但首轮证据里完全没出现"的关键跨度补进检索词
            query = f"{query} {query_extra}"

        # ---- 候选召回：按预算的 4 倍取候选，供证据打分筛选 ----
        n_candidates = max(budget.top_k * 4, budget.top_k + 20)
        candidates = self.candidate_chunks(query, n_candidates,
                                           max_chars=max(budget.max_chars * 3, 60000))

        # ---- 证据打分 / 过滤 ----
        cand_tokens = [tokenize(c.text, use_finance_dict=self.use_finance_dict) for c in candidates]
        idf = idf_weights(cand_tokens)
        ranked = score_evidence_chunks(candidates, features.key_spans,
                                       tokenize(query, use_finance_dict=self.use_finance_dict),
                                       idf, finance_terms=FINANCE_TERMS)
        selected = filter_evidence(ranked, top_k=budget.top_k)

        # ---- 上下文预算：整块累加，超预算丢整块（不做半句截断）----
        budgeted: List[EvidenceChunk] = []
        used_chars = 0
        for ev in selected:
            if used_chars + len(ev.text) > budget.max_chars and budgeted:
                continue
            budgeted.append(ev)
            used_chars += len(ev.text)

        # ---- 记忆压缩 ----
        if use_compression and budgeted:
            # 计算题：前两条证据保留原文（数值/口径上下文一旦被切碎极易算错）
            verbatim_top_n = 2 if features.is_calc else 0
            comp = compress_evidence(
                budgeted, features.key_spans,
                compression_ratio=budget.compression_ratio,
                query_tokens=tokenize(query, use_finance_dict=self.use_finance_dict),
                idf=idf,
                finance_terms=FINANCE_TERMS,
                verbatim_top_n=verbatim_top_n,
            )
            context = comp.context
            comp_stats = comp.stats()
            problems = verify_traceability(comp)
        else:
            comp = compress_evidence(budgeted, features.key_spans, compression_ratio=1.0)
            context = build_context(comp.evidences)
            comp_stats = comp.stats()
            comp_stats["compression_applied"] = False
            problems = verify_traceability(comp)

        chars_before = sum(c.chars_before for c in comp.evidences)
        context_chars = len(context)
        # 压缩比只统计"证据正文"，不把可追溯头部（doc/chunk/page 标注）算进去，
        # 否则未压缩时会出现 ratio > 1 的误导性数字。
        ratio = (sum(len(c.compressed_text) for c in comp.evidences) / chars_before) if chars_before else 1.0

        cov = span_coverage(features.key_spans, context)
        missing = [s.text for s in cov["miss"] if getattr(s, "is_strong", False)][:12]

        # ---- token 估计（有真实 API 数值时由调用方覆盖）----
        prompt_tokens = 0
        if system_prompt or user_prompt:
            prompt_tokens = (estimator or load_estimator()).estimate_prompt(system_prompt, user_prompt)

        selected_info = [
            SelectedChunkInfo(
                doc_id=ce.evidence.doc_id, chunk_id=ce.evidence.chunk_id,
                page=ce.evidence.page, chars_before=ce.chars_before,
                chars_after=ce.chars_after, bm25_score=ce.evidence.bm25_score,
                evidence_score=ce.evidence.evidence_score,
                mode="full" if ce.fallback else "compressed",
                fallback_reason=ce.fallback_reason,
            ) for ce in comp.evidences
        ]

        return PipelineResult(
            qid=qid, domain=self.domain, budget=budget, features=features.to_dict(),
            n_candidates=len(candidates), selected=selected_info, context=context,
            context_chars=context_chars, chars_before_compression=chars_before,
            compression_ratio=ratio, key_span_hit_rate=float(cov["hit_rate"]),
            key_span_missing=missing, prompt_tokens=prompt_tokens,
            use_compression=bool(use_compression), compression_stats=comp_stats,
            traceability_problems=problems, retrieval_seconds=time.time() - t0,
        )


# ---------------------------------------------------------------- Full-text baseline

def build_fulltext_context(doc_ids: Sequence[str], domain: str,
                           page_aware: bool = False,
                           max_chars: Optional[int] = None) -> Tuple[str, Dict[str, object]]:
    """严格全文基线用上下文：直接拼接相关文档全文（不做检索、不做压缩）。

    返回 (context, stats)。stats 里记录原始字符数、是否因超限被跳过。
    """
    parts = []
    chars = 0
    for doc_id in doc_ids:
        if page_aware:
            page_texts = [t for _, t in load_document_pages(doc_id, domain)]
            text = "\n".join(page_texts)
        else:
            text = load_document(doc_id, domain)
        if not text:
            continue
        parts.append(f"[全文 doc={doc_id}]\n{text}")
        chars += len(text)
    context = "\n\n".join(parts)
    if max_chars and len(context) > max_chars:
        context = context[:max_chars]      # 仅用于显式的"截断对照"，默认不用
    stats = {"n_docs": len(parts), "chars": chars, "context_chars": len(context)}
    return context, stats
