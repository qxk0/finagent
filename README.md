# 金融长文档智能问答系统

面向金融长文档的 RAG 问答系统，覆盖保险条款、监管法规、金融合同、上市公司财报、行业研报五个领域：

```
多格式解析 → 分块 → BM25 证据检索 → 动态上下文预算 → 句子级记忆压缩 → Qwen 推理作答
```

重点解决"单篇文档远超模型上下文窗口、全文直入不可行"条件下的**成本可控**与**证据完整性**问题
（实测：A 榜 100 题中 48 题的关联文档全文超出 128K 上下文，平均 73.3 万字符/题）。

> 说明：本 README 的所有数字都来自仓库内的实验产物（`experiments/`），没有 gold label 的指标一律标注
> "待标注"或"估计值"，不做虚构。

## 1. 系统流程

```
题目加载 → 文档解析(PDF/HTML/TXT) → 分块(600字符/100重叠)
        → 动态上下文预算(题目特征 → simple/normal/complex → top_k / max_chars)
        → BM25 候选召回(jieba + 金融术语词典)
        → 证据打分与过滤(BM25 归一 + 关键跨度覆盖 + 查询词覆盖 + 去重/单文档占比限制)
        → 动态记忆压缩(句子级保留，保护数字/百分比/金额/日期/条款号/实体，可追溯 doc_id/chunk_id/page)
        → Qwen 证据推理 → 答案规范化 → answer.csv / answer_b.csv
        → 失败检测 → 定向重检索 → 二次作答（计算题）
```

## 2. 目录结构

```
main.py                       A 榜主流程（题目自带 doc_ids）
main_b.py                     B 榜主流程（全领域检索 + 计算/抽取题）
preprocess.py                 文档解析与分块（含逐页解析，用于 page 级追溯）
retriever.py                  BM25 检索 + jieba 金融词典（可关闭词典做消融）
reasoner.py                   Prompt、Qwen 调用、答案提取、多选自一致性
evaluate.py                   Golden Set 评测（overall/domain/题型/混淆矩阵/失败清单）
fix_b.py                      B 榜计算题定向修复脚本（复用抽象后的失败重检索模块）
config.py                     配置（含三档预算、日志路径、全文基线窗口上限）

agent/                        动态记忆压缩问答核心包
  text_features.py            题目/文本关键跨度（数字·金额·日期·条款号·实体）与题目特征
  context_budget.py           动态上下文预算（三档，参数由实验选出）
  memory_compressor.py        确定性记忆压缩（句子级、可追溯、带 fallback）
  token_estimator.py          在真实 API token 上标定的 prompt token 估计器
  self_consistency.py         多选多次采样 + 提前停止 + 结构化日志
  failure_retrieval.py        首轮 → 失败检测 → 定向重检索 → 二次作答
  pipeline.py                 BM25 → 证据打分 → 动态预算 → 压缩 → 上下文
  corpus.py                   领域语料清单

experiments/                  实验与调参脚本、结果表、日志
tests/                        不变量测试（可追溯性、压缩保护、分档、自一致性）
```

## 3. 快速开始

```bash
pip install -r requirements.txt
set DASHSCOPE_API_KEY=sk-xxx          # Linux/Mac: export DASHSCOPE_API_KEY=sk-xxx

python main.py                        # A 榜：生成 answer.csv
python main_b.py                      # B 榜：生成 answer_b.csv
python fix_b.py                       # B 榜计算题定向修复

python evaluate.py --audit            # 先确认数据里有没有标准答案
python evaluate.py                    # 有标签时评测（无 Key 时可用 --predictions answer.csv 离线打分）

python tests/test_memory_compressor.py        # 记忆压缩不变量测试
python tests/test_self_consistency.py         # 自一致性 + 日志测试

# 离线（不调用 API）：预算/压缩参数选择、token 标定、消融、全文基线可行性
python experiments/tune_budget.py --split A
python experiments/calibrate_token_estimator.py
python experiments/run_ablation.py --split A --limit 100
python experiments/run_fulltext_baseline.py --split A
python experiments/self_consistency_stats.py --infer
python experiments/run_failure_retrieval_demo.py --split B --limit 5

# 真实 API 实验（会消耗 token）
python experiments/run_ablation.py --split A --llm
python experiments/run_fulltext_baseline.py --split A --llm
```

## 4. 关键设计与实验依据

### 4.1 动态上下文预算（任务2）

原来的实现是"每个领域一个固定 top_k / max_chars"。现在改为按题目特征分三档，特征包括：
题面长度、选项中数字/金额/百分比/日期/条款号数量、实体数量、是否多选、是否跨文档、是否需要计算。

参数不是拍的，是 `experiments/tune_budget.py` 在 **200 道真题**上跑"关键跨度覆盖率曲线"选出来的：

* 参考点：top_k=60、不设字符上限
* 选择规则：先在全题池覆盖曲线上找**膝点**（成本优先，取达到参考覆盖率 95% 的最小
  `(max_chars, top_k)`），再按 0.5x / 1x / 1.5x 生成三档
* 得到的锚点：`top_k=20, max_chars=10,000`，该点相对覆盖率 **95.5%**

| 分档 | top_k | max_chars | 压缩目标比 | 关键跨度覆盖率（相对 60 块参考） | 题量(A榜) |
|---|---|---|---|---|---|
| simple | 10 | 4,000 | 0.50 | 86.4% | 16 |
| normal | 20 | 10,000 | 0.60 | 95.5% | 23 |
| complex | 30 | 16,000 | 0.75 | 96.3% | 51 |

（B 榜同样流程跑出的锚点为 `top_k=25, max_chars=16,000`、覆盖率 95.5%，说明"一套固定 top_k 打天下"确实不合理；
结果见 `experiments/budget_tuning_b.json`。）

三档阈值取 200 道题复杂度得分的 p33/p66，实际分档规模 69 / 65 / 66 题，约为 1:1:1。
每题实际使用的 `top_k / max_chars / selected_chunks / context_chars / prompt_tokens` 全部写入
`experiments/logs/retrieval_budget.jsonl`（新流水线）与主流程日志。

### 4.2 动态记忆压缩（任务1）

`agent/memory_compressor.py`，完全确定性、不调用 LLM：

1. 句子切分 → 句级打分（题目关键跨度命中、IDF 加权词覆盖、条款编号、数字密度、位置先验）
2. 命中题目关键跨度（金额/百分比/日期/条款号/多位数字/实体）的句子**优先且必须保留**；
   若保护性句子之和超过该块预算的 1.5 倍，则**整块回退原文**，绝不删关键信息
3. 只做"整句保留/整句丢弃"，**没有任何 `text[:n]` 式截断**
4. 每条证据保留 `doc_id / chunk_id / page（有则记）/ 原文`，上下文头部形如
   `[E3 doc=annual_byd_2024_report|chunk=123|p45|compressed|812/1500字符]`
5. 计算题（计算/抽取）前两条证据强制保留原文（数值口径容易被切碎）
6. `verify_traceability()` 校验"压缩后的每一句都是原文子串、关键跨度无丢失"，
   `tests/test_memory_compressor.py` 与每题日志里都记录校验结果（当前 100 题实测 **0 违规**）

压缩比同样由实验选：在选定预算下扫 `compression_ratio`，取"关键跨度保护率 = 100%、
回退块占比 ≤ 50%、句子保留率 ≥ 40%"的最小压缩比（`experiments/compression_tuning_a.csv`）。

### 4.3 Golden Set：数据里没有标准答案（任务3）

`python evaluate.py --audit` 的结论：

* A/B 榜 200 道题的字段只有 `qid / domain / split / question / options / type / answer_format / doc_ids`，
  **不存在任何答案字段**；
* 数据目录里唯一带答案列的 `upload_b/submit.csv` 是占位值（`999999.99%` 等），被自动标记为
  `PLACEHOLDER_SUSPECT` 并排除；
* 审计结果：`experiments/labels_audit.json`，结论 `NO_GOLD_LABELS_FOUND`。

因此：**当前无法给出任何真实准确率**，也没有伪造标签。仓库提供的是完整评测框架：
自动构建标签（`--build-labels`，仅当数据中确实存在答案时才写文件）、人工标注模板
（`experiments/eval_labels_template.json`）、以及 overall / domain / answer_format /
单选·多选·判断·计算·抽取 分类准确率 + 混淆矩阵 + 失败清单的输出。
拿到标签后一条命令即可产出真实准确率与消融对照。

### 4.4 Full-text 基线（任务4）：先在物理上判断能不能跑

严格基线 = 题目 + **相关完整文档** → Qwen，不用 BM25、不压缩。实测（`experiments/baseline_fulltext_a.csv`）：

| 领域 | 题数 | 平均全文字符数 | 能否塞进 128K 上下文 |
|---|---|---|---|
| 金融合同 | 20 | 879,319 | 0/20 可行 |
| 上市公司财报 | 20 | 788,321 | 0/20 可行 |
| 保险条款 | 20 | 124,921 | 15/20 可行 |
| 监管法规 | 20 | 29,392 | 20/20 可行 |
| 行业研报 | 20 | 109,292 | 17/20 可行 |

**A 榜 100 题中只有 52 题能执行严格全文基线**（其余 48 题全文超出模型上下文窗口，
脚本记为 `SKIPPED_CONTEXT_LIMIT` 并保留实测字符数，而不是靠截断假装跑通）。
B 榜没有 `doc_ids`，领域全量语料 1.0M–5.7M 字符，**0/100 题可行**。

## 5. 实验结果（A 榜 100 题）

来源：`experiments/results.csv`、`experiments/details/a_*.jsonl`、`experiments/ablation_summary.json`。
运行方式：`python experiments/run_ablation.py --split A --limit 100`（离线模式）。

> token 标尺说明：本机没有 API Key（网络受限），因此表中 token 为**估计值**（`token_source=estimated`），
> 估计器在 A 榜真实运行日志的 100 次 API 返回上标定，单次调用 MAPE ≈ **9.3%**、符号偏差均值 +1.3%
> （`experiments/token_estimator.json`、`token_estimator_validation.csv`）。
> **字符数、块数、压缩比、覆盖率、可行性是精确值；所有 token 数字都是估计值。**
> 逐题明细：`experiments/details/a_<组名>.jsonl`（B 榜为 `b_<组名>.jsonl`），可用
> `experiments/run_ablation.py --split A` 离线复现（不消耗 API）。


## 6. 失败驱动重检索（任务6）

`agent/failure_retrieval.py` 把 `fix_b.py` 的硬编码修复逻辑抽象成通用流程：

```
First Pass → failure detection → targeted retrieval → re-answer
```

失败判定只用可观测信号，**不伪造 confidence**：`empty_answer`、`format_violation_choice`、
`format_violation_numeric`、`format_violation_percent`、`format_violation_decimals`、
`numeric_evidence_miss`、`numeric_evidence_low_coverage`、`self_consistency_disagreement`。
重检索会：按题目实体（公司→年报映射）锁定文档、`top_k ×2`、字符上限 ×1.5、
并把"题目里出现但首轮证据中完全没出现"的关键数字补进检索词。
每题记录 `first_answer / first_retrieval_top_k / retry_reason / retry_top_k / second_answer /
additional_tokens / accepted` 到 `experiments/logs/failure_retrieval.jsonl`。

离线验证（`experiments/run_failure_retrieval_demo.json`，用确定性失败桩代替模型输出）：

* 检测器 8 类用例 **8/8 通过**（含"正常答案不应报错"的负例）；
* 5 道 B 榜计算题全部触发重检索：`top_k` 20→40 / 30→60，平均上下文 **+5,239 字符（+63%）**；
  二次答案通过格式验收的有 2/5（桩返回值为占位 `0`，故验收比例不代表真实效果）；
* 这 5 题的关键跨度命中率没有提升（0.67/0.80 → 不变），说明**这类题的瓶颈不在检索而在计算/口径**，
  该结论只有在 gold label 或 `--llm` 重答对照下才能进一步判断——现在不做结论。


## 7. 数据说明

原始数据集（`public_dataset_upload/`、`upload_b/`）不随仓库发布，仓库只包含代码、实验脚本与
实验产物（`experiments/`，均由脚本离线复现）。A 榜 100 题 / B 榜 100 题，合计 5 个领域：

* **A 榜**：68 篇题目自带的关联文档（保险 16 + 监管 15 + 研报 15 + 合同 13 + 年报 9）
* **B 榜**：全领域语料共 **573 篇**，其中 BM25 索引 **443 篇**
  （保险 16 + 合同 14 + 年报 10 + 研报 20 + 监管 383）
* 另有 **130 个监管附件 PDF** 已解析入缓存（历史那批 B 榜运行处理了 513 篇监管文档），
  但当前代码为控制内存占用默认跳过附件（`agent/corpus.py:get_regulatory_doc_ids`）
