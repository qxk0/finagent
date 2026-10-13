"""prompt token 估计器（离线消融实验用，可被真实 API 数值覆盖）。

为什么需要它：动态预算/记忆压缩的消融实验如果每一组都去调一次 API，
成本会成倍上升。因此我们用一个**在真实 API 返回的 prompt_tokens 上标定过**的
线性模型（特征 = 中文/ASCII/其它字符数）来估计离线实验的 token 消耗。

标定脚本：`experiments/calibrate_token_estimator.py`
标定数据：A 榜 100 题真实运行（run_log.txt 的 prompt_tokens + 可精确复现的上下文文本）
标定结果：`experiments/token_estimator.json`（含 MAPE、样本数等）

注意：估计值永远带 `estimated=True` 标记，绝不冒充真实 API 统计。
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import config

from .text_features import char_mix

ESTIMATOR_PATH = os.path.join(config.EXPERIMENTS_DIR, "token_estimator.json")

# 未标定时的兜底模型（仅用于"没有标定文件"的极端情况，误差较大）
_FALLBACK_FEATURES = ["cjk", "ascii_alnum", "other"]
_FALLBACK_COEF = [0.85, 0.35, 0.45]

# 候选特征（注意：chars = cjk + ascii_alnum + other，同时使用会造成共线性，
# 因此通过交叉验证在若干候选模型里选最稳的一个）
FEATURE_BUILDERS = {
    "cjk": lambda mix: float(mix["cjk"]),
    "ascii_alnum": lambda mix: float(mix["ascii_alnum"]),
    "other": lambda mix: float(mix["other"]),
    "chars": lambda mix: float(mix["total"]),
    "one": lambda mix: 1.0,
}

CANDIDATE_MODELS = [
    ["chars", "one"],
    ["cjk", "ascii_alnum", "other"],
    ["chars", "ascii_alnum", "one"],
    ["cjk", "other", "one"],
]


@dataclass
class TokenEstimator:
    features: List[str] = field(default_factory=lambda: list(_FALLBACK_FEATURES))
    coef: List[float] = field(default_factory=lambda: list(_FALLBACK_COEF))
    metrics: Dict[str, object] = field(default_factory=dict)
    source: str = "fallback"

    def estimate(self, text: str) -> int:
        mix = char_mix(text or "")
        val = sum(c * FEATURE_BUILDERS[f](mix) for f, c in zip(self.features, self.coef))
        # 外推保护：token 数不可能偏离字符数太多
        n = mix["total"]
        if n:
            val = min(val, 1.6 * n)
            val = max(val, 0.25 * n)
        return max(0, int(round(val)))

    def estimate_prompt(self, system_prompt: str, user_prompt: str) -> int:
        return self.estimate(system_prompt) + self.estimate(user_prompt)


def load_estimator(path: str = None) -> TokenEstimator:
    path = path or ESTIMATOR_PATH
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features")
            coef = data.get("coef")
            if features is None and isinstance(coef, dict):
                # 兼容早期 {cjk, ascii_alnum, other, intercept} 形式的字典系数
                features = ["cjk", "ascii_alnum", "other", "one"]
                coef = [coef.get(k, 0.0) for k in features]
            if features is None or not isinstance(coef, list):
                raise ValueError("估计器文件缺少 features/coef")
            return TokenEstimator(features=list(features), coef=[float(c) for c in coef],
                                  metrics=data.get("metrics", {}),
                                  source=os.path.basename(path))
        except Exception as e:
            print(f"  [WARN] 读取 token 估计器失败({path}): {e}，使用兜底系数")
    return TokenEstimator()


def _design(texts: Sequence[str], features: Sequence[str]):
    import numpy as np
    rows = []
    for t in texts:
        mix = char_mix(t)
        rows.append([FEATURE_BUILDERS[f](mix) for f in features])
    return np.asarray(rows, dtype=float)


def _metrics(A, y, pred) -> Dict[str, object]:
    import numpy as np
    mape = float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0)) * 100.0)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {
        "mape_percent": round(mape, 3),
        "r2": round(1.0 - ss_res / ss_tot, 4),
        "max_abs_error": int(np.max(np.abs(pred - y))),
        "mean_abs_error": round(float(np.mean(np.abs(pred - y))), 1),
    }


def _nnls(A, y, iters: int = 4000, lr: float = None) -> "np.ndarray":
    """非负最小二乘（投影梯度），保证估计系数单调、正向，避免外推出荒谬数值。"""
    import numpy as np
    n, d = A.shape
    if lr is None:
        norm = float(np.linalg.norm(A, 2) ** 2) or 1.0
        lr = 1.0 / norm
    coef = np.zeros(d)
    for _ in range(iters):
        grad = A.T @ (A @ coef - y)
        coef = np.maximum(0.0, coef - lr * grad)
    return coef


def fit_estimator(texts: Sequence[str], tokens: Sequence[float],
                  features: Sequence[str] = ("cjk", "ascii_alnum", "other")) -> TokenEstimator:
    """非负最小二乘拟合（不依赖 sklearn，用 numpy）。"""
    import numpy as np
    features = list(features)
    A = _design(texts, features)
    y = np.asarray(tokens, dtype=float)
    coef = _nnls(A, y)
    pred = A @ coef
    metrics = _metrics(A, y, pred)
    metrics["n_samples"] = len(y)
    metrics["model_features"] = features
    return TokenEstimator(features=features, coef=[float(c) for c in coef],
                          metrics=metrics, source="fitted")


def select_and_fit(texts: Sequence[str], tokens: Sequence[float],
                   n_folds: int = 5, seed: int = 2026) -> TokenEstimator:
    """在候选模型里做 K 折交叉验证选型，避免共线性导致的退化拟合。

    选择标准：交叉验证 MAPE 最小，且所有系数非负（token/字符 关系必须单调）。
    """
    import numpy as np
    y = np.asarray(tokens, dtype=float)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, min(n_folds, max(2, n // 5)))

    results = []
    for feats in CANDIDATE_MODELS:
        A = _design(texts, feats)
        errs = []
        for f in folds:
            if len(f) == 0 or len(f) == n:
                continue
            train = np.setdiff1d(np.arange(n), f)
            coef = _nnls(A[train], y[train])
            pred = A[f] @ coef
            errs.extend(np.abs(pred - y[f]) / np.maximum(y[f], 1.0))
        if not errs:
            continue
        cv_mape = float(np.mean(errs) * 100.0)
        results.append((cv_mape, feats))

    if not results:
        return fit_estimator(texts, tokens)
    results.sort(key=lambda x: x[0])
    cv_mape, best_feats = results[0]
    est = fit_estimator(texts, tokens, features=best_feats)
    est.metrics["cv_mape_percent"] = round(cv_mape, 3)
    est.metrics["cv_candidates"] = [{"features": f, "cv_mape_percent": round(m, 3)}
                                    for m, f in results]
    return est


def save_estimator(est: TokenEstimator, path: str = None, extra: Optional[dict] = None) -> str:
    path = path or ESTIMATOR_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"features": est.features, "coef": est.coef,
               "metrics": est.metrics, "source": est.source}
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def estimate_prompt_tokens(system_prompt: str, user_prompt: str,
                           estimator: Optional[TokenEstimator] = None) -> int:
    return (estimator or load_estimator()).estimate_prompt(system_prompt, user_prompt)
