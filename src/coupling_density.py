"""耦合密度：把矩阵元分解到实空间/倒空间以便可视化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class KSpaceDistribution:
    """倒空间分布结果。"""

    k_vals: np.ndarray
    weights: np.ndarray


def coupling_density_realspace(
    psi_i: np.ndarray,
    psi_j: np.ndarray,
    dh_dq: np.ndarray,
) -> np.ndarray:
    """计算实空间耦合密度 D(n)。

    定义：
        D(n) = sum_{n'} psi_i*(n) [dH/dQ]_{n n'} psi_j(n')

    性质：
        sum_n D(n) = <psi_i| dH/dQ |psi_j> = g

    返回：
        D: (n_dim,) 复数数组（与 psi 维度一致）
    """
    psi_i = np.asarray(psi_i)
    psi_j = np.asarray(psi_j)
    dh_dq = np.asarray(dh_dq)
    if psi_i.shape != psi_j.shape:
        raise ValueError("psi_i 与 psi_j 形状必须一致")
    if dh_dq.shape != (psi_i.shape[0], psi_i.shape[0]):
        raise ValueError("dh_dq 形状必须为 (n_dim, n_dim)")

    # 每一行的局域贡献：conj(psi_i[n]) * (dh[n,:] @ psi_j)
    local = dh_dq @ psi_j
    return np.conj(psi_i) * local


def group_density_by_cell(density: np.ndarray, n_cells: int, dof_per_cell: int) -> np.ndarray:
    """把密度按超胞格点（cell）聚合，便于画一维链上的空间分布。

    Args:
        density: (n_dim,) 的密度数组（复数）
        n_cells: 原胞数
        dof_per_cell: 每个原胞自由度数（单带=1，SSH=2）
    """
    density = np.asarray(density)
    if density.shape != (n_cells * dof_per_cell,):
        raise ValueError("density 形状必须为 (n_cells * dof_per_cell,)")
    return density.reshape(n_cells, dof_per_cell).sum(axis=1)


def kspace_distribution(
    psi: np.ndarray,
    n_cells: int,
    a: float = 1.0,
    dof_per_cell: Literal[1, 2] = 1,
    shift: bool = True,
) -> KSpaceDistribution:
    """计算态在倒空间上的权重分布 |<k|psi>|^2。

    约定：
    - 对单带链：psi 的长度为 n_cells，使用 numpy FFT（exp(-i 2π m n /N)）。
    - 对 SSH 双原子链：psi 的长度为 2*n_cells，按 (A0,B0,A1,B1,...) 拆分，
      分别 FFT 后将两支权重相加。

    Returns:
        KSpaceDistribution(k_vals, weights)，其中 weights 之和在数值上应接近 1。
    """
    psi = np.asarray(psi, dtype=complex)
    if dof_per_cell == 1:
        if psi.shape != (n_cells,):
            raise ValueError("dof_per_cell=1 时 psi 形状必须为 (n_cells,)")
        coeffs = np.fft.fft(psi) / np.sqrt(n_cells)
        weights = np.abs(coeffs) ** 2
    else:
        if psi.shape != (2 * n_cells,):
            raise ValueError("dof_per_cell=2 时 psi 形状必须为 (2*n_cells,)")
        psi_a = psi[0::2]
        psi_b = psi[1::2]
        coeffs_a = np.fft.fft(psi_a) / np.sqrt(n_cells)
        coeffs_b = np.fft.fft(psi_b) / np.sqrt(n_cells)
        weights = np.abs(coeffs_a) ** 2 + np.abs(coeffs_b) ** 2

    m_vals = np.arange(n_cells, dtype=float)
    k_vals = 2.0 * np.pi * m_vals / (n_cells * a)

    if shift:
        k_vals = np.fft.fftshift(k_vals)
        weights = np.fft.fftshift(weights)
        # 映射到 [-pi/a, pi/a) 方便阅读
        k_vals = (k_vals + np.pi / a) % (2.0 * np.pi / a) - np.pi / a

    return KSpaceDistribution(k_vals=k_vals, weights=weights.astype(float))

