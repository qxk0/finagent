"""动态上下文预算（任务2）：根据题目特征决定 top_k / max_chars。

参数来源说明（重要，避免"拍脑袋参数"）：
  * 兜底默认值写在 `config.BUDGET_TIERS`
  * 真实取值由 `experiments/tune_budget.py` 在 200 道真题上跑"关键跨度覆盖率饱和曲线"
    后写入 `experiments/budget_tiers.json`；本模块优先读取该文件。
  * 覆盖率饱和是**无标签代理目标**（当前数据集没有标准答案），
    一旦拿到 gold label，可用 `experiments/run_ablation.py` 复核该参数选择。
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import config

from .text_features import QuestionFeatures


@dataclass
class ContextBudget:
    """一题最终使用的上下文预算。"""
    tier: str
    top_k: int
    max_chars: int
    compression_ratio: float          # 压缩后目标保留比例（1.0 = 不压缩）
    use_compression: bool = True
    reasons: List[str] = field(default_factory=list)
    features: Dict[str, object] = field(default_factory=dict)
    source: str = "config_default"    # config_default | tuned_file | override

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_TIER_KEYS = ("top_k", "max_chars", "compression_ratio")


def load_tier_params(path: str = None) -> Dict[str, Dict[str, float]]:
    """读取调参结果；文件不存在时回退到 config 默认。"""
    path = path or config.BUDGET_TIERS_FILE
    tiers = {k: dict(v) for k, v in config.BUDGET_TIERS.items()}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                tuned = json.load(f)
            payload = tuned.get("tiers", tuned)
            for tier, params in payload.items():
                if tier in tiers and isinstance(params, dict):
                    for key in _TIER_KEYS:
                        if key in params:
                            tiers[tier][key] = params[key]
            return tiers
        except Exception as e:  # 配置文件损坏不应影响主流程
            print(f"  [WARN] 读取预算调参文件失败({path}): {e}，改用 config 默认值")
    return tiers


def decide_budget(features: QuestionFeatures,
                  tier_params: Optional[Dict[str, Dict[str, float]]] = None,
                  use_compression: bool = None,
                  top_k_override: Optional[int] = None,
                  max_chars_override: Optional[int] = None) -> ContextBudget:
    """由题目特征得到预算；三档 simple / normal / complex。"""
    tiers = tier_params or load_tier_params()
    tier = features.tier if features.tier in tiers else "normal"
    params = tiers[tier]

    top_k = int(params["top_k"])
    max_chars = int(params["max_chars"])
    ratio = float(params["compression_ratio"])

    reasons = [f"tier={tier}", f"complexity={features.complexity_score:.1f}"]
    # 计算题/抽取题：答案依赖精确数值，弱压缩（但仍保留句子级压缩而非截断）
    if features.is_calc:
        ratio = min(0.9, max(ratio, 0.75))
        reasons.append("calc:raise_compression_ratio")
    # 跨文档对比题：需要多个文档各自出现的证据，扩大候选
    if features.cross_doc:
        top_k = max(top_k, int(params["top_k"]))
        reasons.append("cross_doc:keep_top_k")

    source = "tuned_file" if os.path.exists(config.BUDGET_TIERS_FILE) else "config_default"
    if top_k_override:
        top_k = int(top_k_override)
        source = "override"
        reasons.append(f"top_k_override={top_k}")
    if max_chars_override:
        max_chars = int(max_chars_override)
        source = "override"
        reasons.append(f"max_chars_override={max_chars}")

    if use_compression is None:
        use_compression = config.USE_MEMORY_COMPRESSION

    return ContextBudget(
        tier=tier,
        top_k=top_k,
        max_chars=max_chars,
        compression_ratio=ratio,
        use_compression=bool(use_compression),
        reasons=reasons,
        features=features.to_dict(),
        source=source,
    )
