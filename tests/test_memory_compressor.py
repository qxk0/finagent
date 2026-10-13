"""记忆压缩模块的不变量测试（可独立运行，也兼容 pytest）。

直接运行：
    python tests/test_memory_compressor.py

覆盖的不变量：
  1. 可追溯性：压缩后的每一句都必须是原文的子串（不允许改写/半句截断）
  2. 关键信息保护：题目里出现的关键跨度（金额/百分比/日期/条款号）不能在压缩中丢失
  3. 绝不粗暴截断：压缩结果是若干"完整句子"的连接
  4. fallback：过短块、关闭压缩、保护信息超预算时回退原文
  5. 动态预算：200 道真题都能拿到合法分档，且三档参数单调递增
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.context_budget import decide_budget, load_tier_params
from agent.memory_compressor import (EvidenceChunk, compress_evidence, label_in_text,
                                     verify_traceability)
from agent.text_features import extract_question_features, extract_spans, sentence_split


def _mk_chunk(text: str, doc_id="d1", chunk_id=0, page=None, score=1.0) -> EvidenceChunk:
    return EvidenceChunk(doc_id=doc_id, chunk_id=chunk_id, text=text, page=page,
                         bm25_score=score, evidence_score=score)


LONG_CHUNK = (
    "第一条 本产品的保险期间为10年。"
    "第二条 投保人应当在合同生效之日起30日内交纳首期保险费。"
    "第三条 被保险人身故的，本公司按基本保险金额100万元的120%给付身故保险金。"
    "第四条 若被保险人在第5个保单年度内退保，退保费用比例为5%。"
    "本条款未尽事宜，按照本公司相关规定处理，相关规定的解释权归本公司所有。"
    "第五条 现金价值按照保单账户价值扣除退保费用后计算，具体金额以本公司核算为准。"
    "第六条 本保险合同自2025年1月1日起生效。"
)


def test_traceability_and_protection():
    question = "被保险人身故时，若基本保险金额为100万元，第2条规定下应给付多少身故保险金？退保费用比例5%是否适用？"
    spans = extract_spans(question)
    res = compress_evidence([_mk_chunk(LONG_CHUNK)], spans, compression_ratio=0.35,
                            query_tokens=["身故保险金", "退保费用"])
    assert res.n_chunks_before == 1
    assert not verify_traceability(res), f"可追溯性不变量被破坏: {verify_traceability(res)}"
    ce = res.evidences[0]
    for span in spans:
        if label_in_text(span.text, LONG_CHUNK):
            assert label_in_text(span.text, ce.compressed_text), f"关键跨度丢失: {span.text}"


def test_no_mid_sentence_cut():
    spans = extract_spans("利息按照5%计算，期间为30日，依据第三条的规定。")
    res = compress_evidence([_mk_chunk(LONG_CHUNK)], spans, compression_ratio=0.3,
                            query_tokens=["利息", "计算"])
    src_sentences = set(sentence_split(LONG_CHUNK))
    for s in sentence_split(res.evidences[0].compressed_text):
        assert s in src_sentences, f"出现被切碎的句子: {s!r}"


def test_fallbacks():
    spans = extract_spans("退保费用比例是多少？")
    # 过短块 → 不压缩
    short = _mk_chunk("第四条 退保费用比例为5%。")
    res = compress_evidence([short], spans, compression_ratio=0.4)
    assert res.evidences[0].fallback and res.evidences[0].fallback_reason == "chunk_too_short"

    # compression_ratio=1.0 → 明确关闭
    res2 = compress_evidence([_mk_chunk(LONG_CHUNK)], spans, compression_ratio=1.0)
    assert res2.evidences[0].fallback_reason == "compression_disabled"

    # 保护信息超预算 → 回退原文且不丢信息
    # 构造一个"几乎每句都含题目数字"的块：按 5% 压缩必然要丢关键信息，必须回退
    dense = ("第二条 投保人应当在合同生效之日起30日内交纳首期保险费。"
             "第三条 被保险人身故的，本公司按基本保险金额100万元的120%给付身故保险金。"
             "第四条 若被保险人在第5个保单年度内退保，退保费用比例为5%。"
             "第六条 本保险合同自2025年1月1日起生效，保险期间为10年。"
             "第七条 第五条所述现金价值的计算以30日为结算周期。")
    spans3 = extract_spans("30日、100万元的120%、5%、2025年1月1日、10年、第七条，如何计算？")
    res3 = compress_evidence([_mk_chunk(dense)], spans3, compression_ratio=0.05,
                             min_chars_per_chunk=50)
    ce = res3.evidences[0]
    assert ce.fallback, "保护信息超预算时应回退原文"
    assert ce.chars_after == ce.chars_before
    assert not verify_traceability(res3)


def test_budget_tiers():
    qs = []
    for p in sorted(glob.glob(os.path.join(config.QUESTIONS_DIR, "*.json"))):
        qs += json.load(open(p, encoding="utf-8"))
    b_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "upload_b", "question_b")
    if os.path.isdir(b_dir):
        for p in sorted(glob.glob(os.path.join(b_dir, "*"))):
            if p.endswith(".jsonl"):
                qs += [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
            elif p.endswith(".json"):
                qs += json.load(open(p, encoding="utf-8-sig"))
    assert qs, "未找到题目数据"

    tiers = load_tier_params()
    seen = {}
    for q in qs:
        f = extract_question_features(q["question"], q.get("options") or {}, qid=q["qid"],
                                      domain=q["domain"], doc_ids=q.get("doc_ids") or [],
                                      answer_format=q.get("answer_format"), q_type=q.get("type"))
        b = decide_budget(f, tiers)
        assert b.tier in tiers, f"非法分档: {b.tier}"
        assert b.top_k > 0 and b.max_chars > 0 and 0 < b.compression_ratio <= 1.0
        seen[b.tier] = seen.get(b.tier, 0) + 1
    assert set(seen) == {"simple", "normal", "complex"}, f"三档未全部出现: {seen}"
    # 预算阶梯单调
    order = ["simple", "normal", "complex"]
    for a, b in zip(order, order[1:]):
        ta, tb = tiers[a], tiers[b]
        assert (tb["top_k"], tb["max_chars"]) >= (ta["top_k"], ta["max_chars"]), \
            f"预算阶梯非单调: {a}={ta} vs {b}={tb}"
    print("  分档题量:", seen)


def test_real_document_compression():
    """在真实缓存文档上跑一遍，确认没有可追溯性问题。"""
    from preprocess import build_doc_chunks
    chunks = build_doc_chunks("1", "insurance")
    if not chunks:
        print("  [SKIP] 未找到保险文档缓存")
        return
    question = "平安智盈金生在第7个保单年度末退保，退保费用比例和现金价值如何计算？已交保费100万元。"
    spans = extract_spans(question)
    evs = [_mk_chunk(t, doc_id="1", chunk_id=i, score=1.0) for t, i in chunks[:12]]
    res = compress_evidence(evs, spans, compression_ratio=0.4, query_tokens=["退保", "现金价值"])
    problems = verify_traceability(res)
    assert not problems, problems
    print(f"  真实文档压缩: {res.stats()}")


def main():
    tests = [test_traceability_and_protection, test_no_mid_sentence_cut, test_fallbacks,
             test_budget_tiers, test_real_document_compression]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
