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

### 5.1 全量口径：本系统在 100 题上的消耗

| 实验组 | 说明 | 平均上下文字符（100 题） | 平均选中块 | 压缩比 | 总 token（100 题，估计） |
|---|---|---|---|---|---|
| B | BM25（纯 jieba） | 14,096 | 24.0 | 1.00 | 1,131,343 |
| C | BM25 + 金融词典 | 14,083 | 24.0 | 1.00 | 1,139,504 |
| D | C + 动态上下文预算 | 12,541 | 19.9 | 1.00 | 1,134,101 |
| E | D + 动态记忆压缩 | **9,473** | 19.9 | **0.677** | **931,957** |
| F | E + 多选自一致性 | 9,473 | 19.9 | 0.677 | 931,957（离线不采样，与 E 相同；见 §6） |

> A 组（全文基线）无法对齐到 100 题：其中 48 题超出模型上下文窗口被跳过、token 记 0，
> 所以 A 组的"100 题总量"在数值上等于 5.2 的 52 题总量 2,655,117。**不要把 5.1 的
> 100 题总量与 5.2 的 52 题基线直接相除**（例如 1 − 931957/2655117 = 64.90% 是混合了
> 两个不同分母的错误算法）。

### 5.2 同口径对照：全文基线**可执行**的 52 题子集（分子、分母、率同行，可直接验算）

子集构成：监管 20 + 保险 15 + 研报 17 = 52 题。被排除的 48 题 = 全部合同 20 + 全部年报 20
+ 保险 5 + 研报 3（平均 73.3 万字符/题）——即"能跑全文基线的题"恰好都是小文档题。

| 实验组 | 该子集 token（分子） | 基线 token（分母） | 节省率 = 1 − 分子/分母 | 该子集平均每题 token |
|---|---|---|---|---|
| A Full-text 基线 | 2,655,117 | 2,655,117 | 0.00% | 51,059.9 |
| B BM25（纯 jieba） | 542,782 | 2,655,117 | 79.56% | 10,438.1 |
| C BM25 + 金融词典 | 543,096 | 2,655,117 | 79.55% | 10,444.2 |
| D C + 动态上下文预算 | 546,897 | 2,655,117 | 79.40% | 10,517.2 |
| E D + 动态记忆压缩 | **438,719** | 2,655,117 | **83.48%** | 8,436.9 |

### 5.3 全文基线**不可执行**的 48 题（没有分母，因此不计算任何节省率）

| 实验组 | 该 48 题 token（估计） |
|---|---|
| B | 588,561 |
| C | 596,408 |
| D | 587,204 |
| E | 493,238 |

这 48 题若强行全文直入，按同一估计器需要 34,530,564 token，远超 128K 上下文窗口，
属于**物理上不可执行**，因此不存在可比较的基线。

### 5.4 组内同口径对比（两边取同一批题，率可直接验算）

| 对比 | 52 题子集 | 全 100 题 |
|---|---|---|
| E vs C（token） | 1 − 438719/543096 = 19.22% | 1 − 931957/1139504 = 18.21% |
| E vs D（token） | 1 − 438719/546897 = 19.78% | 1 − 931957/1134101 = 17.82% |
| C → E（上下文字符，精确值） | 13,755.1 → 9,115.4（−33.73%） | 14,083.4 → 9,472.7（−32.74%） |

可以得到、且可复现的结论：

1. **动态记忆压缩把上下文从 12,541 字符压到 9,473 字符（−24.5%），压缩比 0.677，
   关键跨度保护率 99.96%–100%，可追溯性校验 0 违规**；
2. **在全文基线可执行的 52 题子集上，系统估计 token 为基线的 16.5%（节省 83.48%）；
   该比例只对这个子集成立，不能外推到 100 题**（其余 48 题没有基线）；系数敏感性检查
   显示该比例在 83.5%–85.5% 区间内稳定，但绝对值仍是估计值；
3. **48/100 题的全文基线在 128K 上下文下物理不可行**——这是"必须做检索/压缩"的最直接证据；
4. 金融词典（B→C）在这批题上对上下文规模影响约 +0.7%，其对**检索质量/准确率**的影响
   必须等 gold label 才能判断，现在不做结论。

### B 榜 100 题（全领域检索、计算题占 26%）

来源：`experiments/results_b.csv`、`experiments/details/b_*.jsonl`（B 榜使用 `--split B` 调出的
预算参数，见 `experiments/budget_tuning_b.json`；表中字符数为取整值，token 为估计值）。

| 实验组 | 平均上下文字符 | 平均选中块 | 压缩比 | 总 token（估计） |
|---|---|---|---|---|
| B BM25（纯 jieba） | 14,124 | — | 1.00 | 1,164,144 |
| C BM25 + 金融词典 | 14,128 | — | 1.00 | 1,171,496 |
| D C + 动态上下文预算 | 17,680 | 28.0 | 1.00 | 1,593,018 |
| E D + 动态记忆压缩 | **13,712** | 28.0 | **0.736** | **1,331,879** |

B 榜的结论与 A 榜**不同，且必须如实说明**：B 榜题目（多选题 66 + 计算题 26）在覆盖率曲线上
需要更大的预算（膝点 `top_k=25 / max_chars=16,000`），所以 D 组比固定预算的 C 组多花约 36% token
换来更高覆盖率；压缩（E）把这部分开销压回 13,712 字符（比 C 少 2.9%，比 D 少 22.4%）。
**这些开销是否值得，只有 gold label 到位后才能判断**，README 不做"更贵但更好"的暗示。
B 榜严格全文基线 0/100 可行（各领域语料 1.0M–5.7M 字符）。

## 6. 多选题自一致性统计（任务5）

自一致性逻辑保持原样（3 次采样、2 票一致提前停止、平票取字母更少者），新增结构化日志字段：
`call_count / candidate_answers / distinct_candidates / early_stop / final_answer / prompt_tokens /
completion_tokens / total_tokens`。日志写入 `experiments/logs/self_consistency.jsonl`，
统计入口 `python experiments/self_consistency_stats.py`。

当前没有新的 API Key 运行，用**历史真实运行日志反推**（方法：该题真实 prompt_tokens ÷ 单次调用
估计 token，标注 `method=inferred`，见 `experiments/self_consistency_stats.json`）：

| 指标 | 值 |
|---|---|
| 多选题数量 | 65 |
| 平均每题调用次数 | 2.10 |
| 提前停止（2 次一致）题数 | 57（87.7%） |
| 调用 3 次题数 | 7（另 1 题反推值不明确） |

> 真实（非反推）统计需运行主流程，`--llm` 模式会直接写入日志。论文/简历里引用时应使用真实日志值。

## 7. 失败驱动重检索（任务6）

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

## 8. 哪些能写进简历 / 哪些不能

**可以写（有真实实验产物支撑）**

* 动态记忆压缩模块：句子级确定性压缩、关键跨度保护率 99.96%–100%、可追溯性 0 违规、
  上下文 **-24.5%**、压缩比 0.677（A 榜 100 题实测）
* 动态上下文预算：基于 200 道真题的覆盖率曲线选出三档参数（膝点 top_k=20/max_chars=10k，
  相对覆盖率 95.5%），题量分布 69/65/66
* 全文基线对照：实测**48/100 题全文超出 128K 上下文，物理不可行**；在"全文基线可执行的
  52 题"子集上，估计 token 节省 **83.48%**（分子 438,719 / 分母 2,655,117；子集口径与
  字符数为精确值，**token 为标定估计值，MAPE 9.3%**）
* 官方提交记录（**非本仓库实验产物**，来自官方提交平台/榜单）：A 榜 100 题实测
  1,673,137 token（prompt 1,644,451 + completion 28,686，见 `answer.csv`）、B 榜 100 题
  2,002,397 token（见 `answer_b.csv`），合计 3,675,534；相对官方参照基线（全文直入，约 299 万，
  同批 100 题）节省约 `1 − 1673137/2990000 = 44.04%`。引用时必须标注"官方口径"，
  且不可与本仓库 52 题子集的 83.48% 混用
* 失败驱动重检索：8 类可观测失败信号 + 定向重检索（平均上下文 +63%），
  全流程日志化、无伪造置信度
* 工程闭环：真实 API token 标定估计器（MAPE 9.3%）、Golden Set 审计、
  不变量测试、逐题可追溯日志

**不能写（缺真实数据支撑）**

* 任何**准确率/F1** 数字（数据没有 gold label，当前无法计算）
* "自一致性提升 x%"、"重检索提升 x% 准确率"（只有上下文规模与触发统计，没有正确率对照）
* 新链路的"真实 token 节省率"（`--llm` 尚未在本机跑过；5.2 的 83.48% 是估计值。
  若已在赛事平台提交过新链路，以平台的官方数值为准，并且不要与 83.48% 并列）
* "金融词典增强带来 x% 提升"（当前只测到上下文规模 +0.7%）

## 9. 数据说明

原始数据集（`public_dataset_upload/`、`upload_b/`）不随仓库发布，仓库只包含代码、实验脚本与
实验产物（`experiments/`，均由脚本离线复现）。A 榜 100 题 / B 榜 100 题，合计 5 个领域：

* **A 榜**：68 篇题目自带的关联文档（保险 16 + 监管 15 + 研报 15 + 合同 13 + 年报 9）
* **B 榜**：全领域语料共 **573 篇**，其中 BM25 索引 **443 篇**
  （保险 16 + 合同 14 + 年报 10 + 研报 20 + 监管 383）
* 另有 **130 个监管附件 PDF** 已解析入缓存（历史那批 B 榜运行处理了 513 篇监管文档），
  但当前代码为控制内存占用默认跳过附件（`agent/corpus.py:get_regulatory_doc_ids`）
