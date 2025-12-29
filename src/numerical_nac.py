"""数值 NAC 计算工具。"""

from __future__ import annotations

import numpy as np


def align_phase(psi_ref: np.ndarray, psi_target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """相位对齐：最大化 Re⟨ref|target⟩。"""
    overlap = np.vdot(psi_ref, psi_target)
    if np.abs(overlap) < eps:
        return psi_target
    phase = overlap / np.abs(overlap)
    return psi_target * np.conj(phase)


def nac_finite_diff(psi_plus: np.ndarray, psi_minus: np.ndarray, delta: float) -> np.ndarray:
    """有限差分：返回 d|psi>/dQ。"""
    if delta == 0.0:
        raise ValueError("delta 不能为 0")
    return (psi_plus - psi_minus) / (2.0 * delta)


def nac_hellmann_feynman(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    h_plus: np.ndarray,
    h_minus: np.ndarray,
    delta_e: float,
    delta: float,
) -> complex:
    """HF 定理法：⟨psi_i| (H+ - H-) |psi_j⟩ / (2 delta ΔE)。"""
    if delta == 0.0:
        raise ValueError("delta 不能为 0")
    if delta_e == 0.0:
        raise ValueError("delta_e 不能为 0")
    dh = (h_plus - h_minus) / (2.0 * delta)
    return np.vdot(psi_i, dh @ psi_j) / delta_e
