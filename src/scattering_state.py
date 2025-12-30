"""缺陷体系中的散射态选择与 Bloch 特征诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .numerical_nac import align_phase


@dataclass(frozen=True)
class ScatteringStateSelection:
    """散射态选择结果。"""

    idx: int
    energy: float
    overlap: float
    state: np.ndarray


def select_scattering_state(
    evals: np.ndarray,
    evecs: np.ndarray,
    psi_ref: np.ndarray,
    energy_window: Optional[float] = None,
    energy_ref: Optional[float] = None,
    allowed_indices: Optional[np.ndarray] = None,
    energy_tol: float = 1e-6,
) -> ScatteringStateSelection:
    """从含缺陷本征态中选择与参考态重叠最大的散射态。

    说明：
    - 若候选态存在简并/近简并，本函数会在简并子空间内做投影重构，
      以得到与参考态最匹配的线性组合（避免“任意驻波基”问题）。
    - 返回的 state 会做全局相位对齐，使得 Re⟨ref|state⟩ 最大。

    Args:
        evals: 含缺陷哈密顿量本征值 (n_state,)
        evecs: 含缺陷哈密顿量本征矢（列向量）(n_dim, n_state)
        psi_ref: 参考态（通常为纯净体系 Bloch 态）(n_dim,)
        energy_window: 可选能窗约束 |E_n - energy_ref| <= energy_window
        energy_ref: 能窗中心能量；未指定时使用“最大重叠态”的能量
        allowed_indices: 允许的态索引集合（用于排除局域态等）
        energy_tol: 简并判定阈值，用于重构简并子空间线性组合

    Returns:
        ScatteringStateSelection: 含 idx/energy/overlap/state
    """
    evals = np.asarray(evals, dtype=float)
    evecs = np.asarray(evecs)
    psi_ref = np.asarray(psi_ref)

    if evecs.ndim != 2:
        raise ValueError("evecs 必须为二维数组 (n_dim, n_state)")
    if evals.ndim != 1 or evals.shape[0] != evecs.shape[1]:
        raise ValueError("evals 形状必须为 (n_state,) 且与 evecs 列数一致")
    if psi_ref.shape != (evecs.shape[0],):
        raise ValueError("psi_ref 形状必须为 (n_dim,)")

    overlaps = evecs.conj().T @ psi_ref
    weights = np.abs(overlaps)

    if allowed_indices is not None:
        mask = np.zeros_like(weights, dtype=bool)
        allowed_indices = np.asarray(allowed_indices, dtype=int)
        mask[allowed_indices] = True
        weights = np.where(mask, weights, 0.0)

    idx_best = int(np.argmax(weights))
    if weights[idx_best] == 0.0:
        raise ValueError("allowed_indices 过滤后没有可用候选态")

    if energy_ref is None:
        energy_ref = float(evals[idx_best])

    if energy_window is not None:
        if energy_window <= 0:
            raise ValueError("energy_window 必须为正数或 None")
        in_window = np.where(np.abs(evals - float(energy_ref)) <= float(energy_window))[0]
        if allowed_indices is not None:
            in_window = np.intersect1d(in_window, allowed_indices, assume_unique=False)
        if in_window.size == 0:
            raise ValueError("能窗内没有可用候选态，请调整 energy_window/energy_ref")
        idx_best = int(in_window[np.argmax(np.abs(overlaps[in_window]))])

    energy_center = float(evals[idx_best])
    close = np.where(np.abs(evals - energy_center) <= float(energy_tol))[0]
    if allowed_indices is not None:
        close = np.intersect1d(close, allowed_indices, assume_unique=False)

    if close.size <= 1:
        psi = evecs[:, idx_best]
    else:
        proj = evecs[:, close] @ overlaps[close]
        norm = np.linalg.norm(proj)
        if norm == 0:
            psi = evecs[:, idx_best]
        else:
            psi = proj / norm

    psi = align_phase(psi_ref, psi)
    overlap = float(np.abs(np.vdot(psi, psi_ref)))

    return ScatteringStateSelection(
        idx=idx_best,
        energy=float(evals[idx_best]),
        overlap=overlap,
        state=psi,
    )


def max_overlap_with_reference(
    evecs: np.ndarray,
    psi_ref: np.ndarray,
    allowed_indices: Optional[np.ndarray] = None,
) -> ScatteringStateSelection:
    """仅计算"最大重叠态"（不做能窗/简并投影），用于快速诊断 Bloch 特征。"""
    evecs = np.asarray(evecs)
    psi_ref = np.asarray(psi_ref)
    overlaps = evecs.conj().T @ psi_ref
    weights = np.abs(overlaps)
    if allowed_indices is not None:
        mask = np.zeros_like(weights, dtype=bool)
        allowed_indices = np.asarray(allowed_indices, dtype=int)
        mask[allowed_indices] = True
        weights = np.where(mask, weights, 0.0)
    idx = int(np.argmax(weights))
    if weights[idx] == 0.0:
        raise ValueError("allowed_indices 过滤后没有可用候选态")
    psi = align_phase(psi_ref, evecs[:, idx])
    return ScatteringStateSelection(
        idx=idx,
        energy=float("nan"),
        overlap=float(weights[idx]),
        state=psi,
    )

