"""动态记忆压缩（Dynamic Memory Compression，任务1）。

设计约束（面向金融长文档问答的压缩需求）：
  1. **不做字符串粗暴截断**：压缩的最小单位是句子，只做"整句保留/整句丢弃"。
  2. 优先保留：数字、百分比、金额、日期、条款编号、公司/产品实体、与问题直接相关的句子。
     这些信息被定义为"关键跨度（protected spans）"，命中它们的句子会被优先保留；
     若保护性句子之和超出该块预算，宁可整块回退原文，也不丢保护信息。
  3. 每条压缩证据保留 doc_id / chunk_id / page（若有）/ 原文，可完整追溯。
  4. 完全不调用 LLM：全部是确定性规则 + 打分（jieba 只用于分词）。
  5. 允许 fallback：计算题的前两条证据默认保留原文；压缩收益不足时也不压缩。
"""
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .text_features import Span, _normalize, sentence_split, span_in_text


# ------------------------------------------------------------------ 数据结构

@dataclass
class EvidenceChunk:
    """一条候选证据（来自 BM25 的原始 chunk）。"""
    doc_id: str
    chunk_id: int
    text: str
    page: Optional[int] = None
    bm25_score: float = 0.0
    source: str = "bm25"          # bm25 | targeted | whole_doc
    evidence_score: float = 0.0   # 证据打分（BM25 归一 + 关键跨度覆盖 + 词覆盖）
    rank: int = 0

    @property
    def key(self) -> str:
        return f"{self.doc_id}#{self.chunk_id}"

    @property
    def citation(self) -> str:
        page = f"|p{self.page}" if self.page else ""
        return f"doc={self.doc_id}|chunk={self.chunk_id}{page}"


@dataclass
class CompressedEvidence:
    evidence: EvidenceChunk
    compressed_text: str
    kept_sentence_idx: List[int] = field(default_factory=list)
    n_sentences: int = 0
    protected_found: List[str] = field(default_factory=list)
    protected_kept: List[str] = field(default_factory=list)
    evidence_id: str = ""
    fallback: bool = False
    fallback_reason: str = ""

    @property
    def chars_before(self) -> int:
        return len(self.evidence.text)

    @property
    def chars_after(self) -> int:
        return len(self.compressed_text)

    @property
    def ratio(self) -> float:
        return self.chars_after / self.chars_before if self.chars_before else 1.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "doc_id": self.evidence.doc_id,
            "chunk_id": self.evidence.chunk_id,
            "page": self.evidence.page,
            "source": self.evidence.source,
            "bm25_score": round(self.evidence.bm25_score, 4),
            "evidence_score": round(self.evidence.evidence_score, 4),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "n_sentences": self.n_sentences,
            "kept_sentence_idx": list(self.kept_sentence_idx),
            "protected_found": list(self.protected_found),
            "protected_kept": list(self.protected_kept),
            "fallback": self.fallback,
            "fallback_reason": self.fallback_reason,
            "source_text": self.evidence.text,
            "compressed_text": self.compressed_text,
        }


@dataclass
class CompressionResult:
    evidences: List[CompressedEvidence]
    context: str
    chars_before: int
    chars_after: int
    n_sentences_before: int
    n_sentences_after: int
    protected_total: int
    protected_kept: int
    fallback_chunks: int = 0
    compression_applied: bool = True
    notes: List[str] = field(default_factory=list)

    @property
    def n_chunks_before(self) -> int:
        return len(self.evidences)

    @property
    def n_chunks_after(self) -> int:
        """压缩后仍进入上下文的证据条数（本方案不因压缩而丢证据）。"""
        return sum(1 for e in self.evidences if e.compressed_text.strip())

    @property
    def compression_ratio(self) -> float:
        return self.chars_after / self.chars_before if self.chars_before else 1.0

    @property
    def protected_keep_rate(self) -> float:
        return self.protected_kept / self.protected_total if self.protected_total else 1.0

    def audit_records(self) -> List[Dict[str, object]]:
        """逐条证据的审计记录（含原文与压缩文），用于追溯与人工复核。"""
        return [e.to_dict() for e in self.evidences]

    def stats(self) -> Dict[str, object]:
        return {
            "n_chunks_before": self.n_chunks_before,
            "n_chunks_after": self.n_chunks_after,
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "compression_ratio": round(self.compression_ratio, 4),
            "n_sentences_before": self.n_sentences_before,
            "n_sentences_after": self.n_sentences_after,
            "protected_total": self.protected_total,
            "protected_kept": self.protected_kept,
            "protected_keep_rate": round(self.protected_keep_rate, 4),
            "fallback_chunks": self.fallback_chunks,
            "compression_applied": self.compression_applied,
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------ 打分

_CLAUSE_HEAD_RE = re.compile(r"第\s*[一二三四五六七八九十百千零〇两0-9]+\s*(?:条|章|节|款)")
_NUMERIC_RE = re.compile(r"\d")


def _numeric_core(s: str) -> str:
    """取出跨度里的纯数字核心（"100万元" -> "100"，"12.34%" -> "12.34"）。"""
    return re.sub(r"[^0-9.]", "", s or "")


def label_in_text(label: str, text: str) -> bool:
    """判断某个关键跨度的信息是否仍保留在文本里（单位无关）。"""
    hay = _normalize(text)
    lab = _normalize(label)
    if lab and lab in hay:
        return True
    core = _numeric_core(lab)
    if core and len(core) >= 2 and core in hay:
        return True
    return False


def _sentence_score(sent: str, query_tokens: set, idf: Dict[str, float],
                    protected: Sequence[Span], term_tokens: Sequence[str],
                    position: int, n_sentences: int) -> Tuple[float, List[str]]:
    """给句子打分，并返回命中的保护跨度。"""
    from .text_features import extract_spans

    matched: List[str] = []
    protect_bonus = 0.0
    for sp in protected:
        if sp.norm and span_in_text(sp, sent):
            matched.append(sp.text)
            protect_bonus += 1.2 if sp.is_strong else 0.4

    # 词覆盖（IDF 加权）
    toks = [t for t in re.split(r"[\s，。；、：:（）()【】\[\]％%]+", sent) if t]
    overlap = 0.0
    if query_tokens:
        hit = 0.0
        for t in set(toks):
            if t in query_tokens:
                hit += idf.get(t, 1.0)
        total = sum(idf.get(t, 1.0) for t in set(query_tokens)) or 1.0
        overlap = min(1.0, hit / total) * 3.0

    term_bonus = sum(0.6 for t in term_tokens if t and t in sent)
    clause_bonus = 0.5 if _CLAUSE_HEAD_RE.search(sent) else 0.0
    numeric_density = min(1.0, len(_NUMERIC_RE.findall(sent)) / 12.0)
    position_bonus = 0.3 if position == 0 else (0.15 if position < 2 else 0.0)
    length_penalty = 0.0
    if len(sent) < 8:
        length_penalty = -0.4
    elif len(sent) > 400:
        length_penalty = -0.3

    score = (protect_bonus + overlap + term_bonus + clause_bonus
             + 0.6 * numeric_density + position_bonus + length_penalty)
    return score, matched


def score_evidence_chunks(evidences: Sequence[EvidenceChunk],
                          key_spans: Sequence[Span],
                          query_tokens: Sequence[str],
                          idf: Dict[str, float],
                          finance_terms: Sequence[str] = ()) -> List[EvidenceChunk]:
    """证据打分 + 过滤前的排序依据。

    evidence_score = 0.5 * 归一化BM25 + 0.3 * 关键跨度覆盖率 + 0.2 * 查询词覆盖率
    （关键跨度覆盖率是金融题里最有区分度的信号：题目里的数字/条款号是否真的出现在该块）
    """
    if not evidences:
        return []
    max_bm25 = max((e.bm25_score for e in evidences), default=0.0) or 1.0
    qset = set(query_tokens)

    for ev in evidences:
        strong = [s for s in key_spans if s.is_strong]
        pool = strong or list(key_spans)
        hit = [s for s in pool if s.norm and span_in_text(s, ev.text)]
        span_cov = len(hit) / len(pool) if pool else 0.0

        chunk_tokens = set(re.split(r"[\s，。；、：:（）()【】\[\]％%]+", ev.text))
        term_cov = len(qset & chunk_tokens) / len(qset) if qset else 0.0
        term_bonus = min(1.0, sum(1 for t in finance_terms if t in ev.text) / 3.0)

        ev.evidence_score = (0.5 * (ev.bm25_score / max_bm25)
                             + 0.3 * span_cov
                             + 0.2 * term_cov
                             + 0.05 * term_bonus)

    return sorted(evidences, key=lambda e: -e.evidence_score)


def filter_evidence(evidences: Sequence[EvidenceChunk], top_k: int,
                    min_score: float = 0.06, max_per_doc_ratio: float = 0.8,
                    dedup_jaccard: float = 0.92) -> List[EvidenceChunk]:
    """证据过滤：分数门槛 + 近重复去重 + 单文档占比限制（保证跨文档题的证据多样）。"""
    kept: List[EvidenceChunk] = []
    seen_norm: List[str] = []
    doc_count: Dict[str, int] = {}
    doc_limit = max(1, int(math.ceil(top_k * max_per_doc_ratio)))

    for ev in evidences:
        if ev.evidence_score < min_score and kept:
            continue
        if doc_count.get(ev.doc_id, 0) >= doc_limit and len(kept) >= max(3, top_k // 2):
            continue
        norm = _normalize(ev.text[:400])
        dup = False
        for prev in seen_norm:
            if _jaccard(norm, prev) >= dedup_jaccard:
                dup = True
                break
        if dup:
            continue
        seen_norm.append(norm)
        doc_count[ev.doc_id] = doc_count.get(ev.doc_id, 0) + 1
        kept.append(ev)
        if len(kept) >= top_k:
            break
    return kept


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ------------------------------------------------------------------ 压缩主函数

def compress_evidence(
    evidences: Sequence[EvidenceChunk],
    key_spans: Sequence[Span],
    compression_ratio: float = 0.5,
    query_tokens: Sequence[str] = (),
    idf: Optional[Dict[str, float]] = None,
    finance_terms: Sequence[str] = (),
    min_chars_per_chunk: int = 160,
    max_chars_per_chunk: int = 1400,
    verbatim_top_n: int = 0,
    total_char_budget: Optional[int] = None,
) -> CompressionResult:
    """对候选证据做句子级记忆压缩，返回可追溯的压缩结果。

    Args:
        compression_ratio: 目标保留比例（0.4 = 压缩到 40% 字符）。整篇预算按此计算，
            单块预算按 evidence_score 加权分配。
        verbatim_top_n: 前 N 条证据强制保留原文（计算题用，避免数值上下文被切碎）。
        total_char_budget: 显式给定压缩后的总字符预算（覆盖 ratio 计算）。
    """
    idf = idf or {}
    evidences = list(evidences)
    chars_before = sum(len(e.text) for e in evidences)
    n_sent_before = sum(len(sentence_split(e.text)) for e in evidences)

    if not evidences or compression_ratio >= 0.999:
        kept = [_keep_whole(e, "compression_disabled") for e in evidences]
        return _finalize(kept, chars_before, n_sent_before, applied=False,
                         notes=["compression_disabled"])

    budget_total = int(total_char_budget) if total_char_budget else int(chars_before * compression_ratio)
    budget_total = max(budget_total, len(evidences) * min_chars_per_chunk)

    # 按 evidence_score 加权分配单块预算（分数高 → 保留更多）
    weights = [max(0.05, e.evidence_score) for e in evidences]
    wsum = sum(weights) or 1.0

    out: List[CompressedEvidence] = []
    for rank, ev in enumerate(evidences):
        if rank < verbatim_top_n:
            out.append(_keep_whole(ev, "calc_verbatim"))
            continue

        chunk_budget = int(budget_total * weights[rank] / wsum)
        chunk_budget = max(min_chars_per_chunk, chunk_budget)
        chunk_budget = min(max_chars_per_chunk, chunk_budget, len(ev.text))

        if len(ev.text) <= min_chars_per_chunk * 1.5:
            out.append(_keep_whole(ev, "chunk_too_short"))
            continue
        out.append(_compress_one(ev, key_spans, chunk_budget, query_tokens, idf,
                                 finance_terms, min_chars_per_chunk))

    # 校验：任何原文里出现的关键跨度，压缩后都不能丢；丢了就整块回退原文
    for ce in out:
        missing = [t for t in ce.protected_found
                   if t and not label_in_text(t, ce.compressed_text)]
        if missing:
            whole = _keep_whole(ce.evidence, "protected_span_lost_repair")
            ce.compressed_text = whole.compressed_text
            ce.kept_sentence_idx = whole.kept_sentence_idx
            ce.fallback = True
            ce.fallback_reason = "protected_span_lost_repair"
            ce.protected_kept = list(ce.protected_found)

    return _finalize(out, chars_before, n_sent_before, applied=True, notes=[])


def _keep_whole(ev: EvidenceChunk, reason: str) -> CompressedEvidence:
    sents = sentence_split(ev.text)
    return CompressedEvidence(
        evidence=ev,
        compressed_text=ev.text,
        kept_sentence_idx=list(range(len(sents))),
        n_sentences=len(sents),
        fallback=True,
        fallback_reason=reason,
    )


def _compress_one(ev: EvidenceChunk, key_spans: Sequence[Span], budget: int,
                  query_tokens: Sequence[str], idf: Dict[str, float],
                  finance_terms: Sequence[str], min_chars: int) -> CompressedEvidence:
    from .text_features import extract_spans

    sents = sentence_split(ev.text)
    if len(sents) <= 1:
        return _keep_whole(ev, "single_sentence")

    protected = [sp for sp in key_spans if sp.norm and span_in_text(sp, ev.text)]
    query_set = set(query_tokens)

    scored: List[Tuple[int, float, List[str]]] = []
    for i, s in enumerate(sents):
        sc, matched = _sentence_score(s, query_set, idf, protected, finance_terms,
                                      position=i, n_sentences=len(sents))
        scored.append((i, sc, matched))

    protected_idx = [i for i, sc, matched in scored if matched]
    protected_chars = sum(len(sents[i]) for i in protected_idx)

    # 保护性句子本身已超出预算 1.5 倍 → 整块回退（宁可多给 token，也不丢关键信息）
    if protected_chars > budget * 1.5 and protected_chars > min_chars:
        ce = _keep_whole(ev, "protected_spans_exceed_budget")
        ce.protected_found = [m for _, _, ms in scored for m in ms]
        ce.protected_kept = list(ce.protected_found)
        return ce

    # 先放保护句（按分数降序，但输出时按原顺序）
    selected: List[int] = []
    used = 0
    for i, sc, matched in sorted([t for t in scored if t[2]], key=lambda t: -t[1]):
        selected.append(i)
        used += len(sents[i])
        if used >= budget:
            break

    # 再用剩余预算补高分的普通句
    for i, sc, matched in sorted(scored, key=lambda t: (-t[1], t[0])):
        if i in selected:
            continue
        if used + len(sents[i]) > budget:
            continue
        selected.append(i)
        used += len(sents[i])

    if not selected:
        return _keep_whole(ev, "no_sentence_selected")

    selected.sort()
    text = "\n".join(sents[i] for i in selected)
    found = [m for _, _, ms in scored for m in ms]
    kept = [m for i, _, ms in scored if i in selected for m in ms]
    ce = CompressedEvidence(
        evidence=ev,
        compressed_text=text,
        kept_sentence_idx=selected,
        n_sentences=len(sents),
        protected_found=found,
        protected_kept=kept,
    )
    if len(text) >= len(ev.text):
        ce.fallback = True
        ce.fallback_reason = "no_gain"
    return ce


def _finalize(kept: List[CompressedEvidence], chars_before: int, n_sent_before: int,
              applied: bool, notes: List[str]) -> CompressionResult:
    for n, ce in enumerate(kept, 1):
        ce.evidence_id = f"E{n}"
    context = build_context(kept)
    return CompressionResult(
        evidences=kept,
        context=context,
        chars_before=chars_before,
        chars_after=sum(len(c.compressed_text) for c in kept),
        n_sentences_before=n_sent_before,
        n_sentences_after=sum(len(c.kept_sentence_idx) for c in kept),
        protected_total=sum(len(c.protected_found) for c in kept),
        protected_kept=sum(len(c.protected_kept) for c in kept),
        fallback_chunks=sum(1 for c in kept if c.fallback),
        compression_applied=applied,
        notes=notes,
    )


# ------------------------------------------------------------------ 上下文与追溯

def build_context(evidences: Sequence[CompressedEvidence]) -> str:
    """组装压缩上下文，每条证据都带可追溯的 doc_id / chunk_id / page 与模式标记。"""
    parts = []
    for ce in evidences:
        if not ce.compressed_text.strip():
            continue
        mode = "full" if ce.fallback else "compressed"
        header = (f"[{ce.evidence_id} {ce.evidence.citation}|{mode}"
                  f"|{ce.chars_after}/{ce.chars_before}字符]")
        parts.append(f"{header}\n{ce.compressed_text}")
    return "\n\n".join(parts)


def verify_traceability(result: CompressionResult) -> List[str]:
    """可追溯性自检：压缩文中每一句都必须是原文的子串（不允许改写/截断）。"""
    problems: List[str] = []
    for ce in result.evidences:
        src_norm = _normalize(ce.evidence.text)
        for s in sentence_split(ce.compressed_text):
            if _normalize(s) not in src_norm:
                problems.append(f"{ce.evidence.key}: 句子不在原文中 -> {s[:40]}")
        for span in ce.protected_kept:
            if not label_in_text(span, ce.compressed_text):
                problems.append(f"{ce.evidence.key}: 关键跨度丢失 -> {span}")
    return problems


def trace_quotation(result: CompressionResult, quotation: str) -> List[Dict[str, object]]:
    """给定答案里的一段引文，定位它来自哪些 chunk（用于人工复核/面试演示）。"""
    q = _normalize(quotation)
    hits = []
    for ce in result.evidences:
        if q and q in _normalize(ce.evidence.text):
            hits.append({
                "evidence_id": ce.evidence_id,
                "doc_id": ce.evidence.doc_id,
                "chunk_id": ce.evidence.chunk_id,
                "page": ce.evidence.page,
                "in_compressed_context": q in _normalize(ce.compressed_text),
            })
    return hits
