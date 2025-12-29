"""诊断工具：IPR、选态与收敛性检查。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def ipr(state: np.ndarray) -> float:
    """计算逆参与比 IPR = sum |psi_i|^4。"""
    return float(np.sum(np.abs(state) ** 4))


def find_localized_states(
    eigenvectors: np.ndarray,
    ipr_threshold: float,
) -> np.ndarray:
    """根据 IPR 阈值筛选局域态索引。"""
    ipr_vals = np.array([ipr(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])])
    return np.where(ipr_vals >= ipr_threshold)[0]


def select_bloch_states(
    n_cells: int,
    r1: float,
    r2: float,
    a: float = 1.0,
) -> Tuple[float, float, int, int]:
    """根据有理分数 r1,r2 选择 Bloch 态的 k 点。"""
    m1 = r1 * n_cells
    m2 = r2 * n_cells
    m1_int = int(round(m1))
    m2_int = int(round(m2))
    if not (np.isclose(m1, m1_int) and np.isclose(m2, m2_int)):
        raise ValueError("r1, r2 与 n_cells 不匹配，无法得到整数动量指标")
    k1 = 2.0 * np.pi * m1_int / (n_cells * a)
    k2 = 2.0 * np.pi * m2_int / (n_cells * a)
    return k1, k2, m1_int, m2_int


def select_k_from_ratio(
    n_cells: int,
    ratio: float,
    a: float = 1.0,
) -> Tuple[float, int]:
    """根据有理分数 ratio 选择单个 k 点。"""
    m_val = ratio * n_cells
    m_int = int(round(m_val))
    if not np.isclose(m_val, m_int):
        raise ValueError("ratio 与 n_cells 不匹配，无法得到整数动量指标")
    k_val = 2.0 * np.pi * m_int / (n_cells * a)
    return k_val, m_int


def top_ipr_indices(eigenvectors: np.ndarray, count: int = 2) -> np.ndarray:
    """返回 IPR 最大的若干态索引。"""
    if count <= 0:
        raise ValueError("count 必须为正整数")
    ipr_vals = np.array([ipr(eigenvectors[:, i]) for i in range(eigenvectors.shape[1])])
    order = np.argsort(ipr_vals)[::-1]
    return order[:count]


def find_vbm_cbm_indices(eigenvalues: np.ndarray) -> Tuple[int, int]:
    """在能谱中寻找 VBM/CBM 的索引（假定费米能级在 0）。"""
    negative = np.where(eigenvalues <= 0.0)[0]
    positive = np.where(eigenvalues >= 0.0)[0]
    if negative.size == 0 or positive.size == 0:
        raise ValueError("能谱未跨过 0，无法定义 VBM/CBM")
    vbm_idx = negative[np.argmax(eigenvalues[negative])]
    cbm_idx = positive[np.argmin(eigenvalues[positive])]
    return int(vbm_idx), int(cbm_idx)


def energy_gap(eigenvalues: np.ndarray) -> float:
    """返回 VBM-CBM 能隙。"""
    vbm_idx, cbm_idx = find_vbm_cbm_indices(eigenvalues)
    return float(eigenvalues[cbm_idx] - eigenvalues[vbm_idx])
