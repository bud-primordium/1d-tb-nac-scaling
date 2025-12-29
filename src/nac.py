"""NAC（非绝热耦合）模态热平均计算。"""

from __future__ import annotations

from typing import Literal

import numpy as np

BOLTZMANN_EV = 8.617333262e-5


def qdot_variance(
    omegas: np.ndarray,
    temperature: float,
    mode: Literal["classical", "quantum"] = "classical",
) -> np.ndarray:
    """返回 ⟨Qdot^2⟩（质量加权坐标）。"""
    if mode == "classical":
        return np.full_like(omegas, BOLTZMANN_EV * temperature, dtype=float)
    if mode == "quantum":
        if temperature <= 0.0:
            return omegas * 0.5
        exponent = omegas / (BOLTZMANN_EV * temperature)
        n_q = 1.0 / (np.exp(exponent) - 1.0)
        return omegas * (n_q + 0.5)
    raise ValueError("mode 必须为 'classical' 或 'quantum'")


def mean_square_nac(
    g_qnu: np.ndarray,
    delta_e: float,
    qdot_var: np.ndarray,
) -> float:
    """计算 ⟨|d_ij|^2⟩。"""
    if g_qnu.shape != qdot_var.shape:
        raise ValueError("g_qnu 与 qdot_var 形状必须一致")
    denom = float(delta_e) ** 2
    return float(np.sum(np.abs(g_qnu) ** 2 * qdot_var) / denom)
