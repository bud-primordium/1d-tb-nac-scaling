"""双原子链声子：色散与极化矢量。"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def diatomic_dynamical_matrix(
    q: float,
    k_spring: float,
    mass_a: float,
    mass_b: float,
    a: float = 1.0,
) -> np.ndarray:
    """构造双原子链的质量加权动力学矩阵（2x2）。"""
    cos_term = np.cos(0.5 * q * a)
    d_aa = 2.0 * k_spring / mass_a
    d_bb = 2.0 * k_spring / mass_b
    d_ab = -2.0 * k_spring * cos_term / np.sqrt(mass_a * mass_b)
    d = np.array([[d_aa, d_ab], [d_ab, d_bb]], dtype=float)
    return d


def diatomic_modes(
    q: float,
    k_spring: float,
    mass_a: float,
    mass_b: float,
    a: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回频率与极化矢量（质量加权坐标）。"""
    d = diatomic_dynamical_matrix(q, k_spring, mass_a, mass_b, a=a)
    eigvals, eigvecs = np.linalg.eigh(d)
    eigvals = np.clip(eigvals, 0.0, None)
    omegas = np.sqrt(eigvals)
    return omegas, eigvecs


def diatomic_modes_grid(
    q_vals: np.ndarray,
    k_spring: float,
    mass_a: float,
    mass_b: float,
    a: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """批量计算双原子链的频率与极化矢量。"""
    omegas = np.zeros((q_vals.shape[0], 2), dtype=float)
    evecs = np.zeros((q_vals.shape[0], 2, 2), dtype=float)
    for idx, q in enumerate(q_vals):
        omega_q, evec_q = diatomic_modes(q, k_spring, mass_a, mass_b, a=a)
        omegas[idx] = omega_q
        evecs[idx] = evec_q
    return omegas, evecs


def displacement_from_q(
    q_vals: np.ndarray,
    q_amplitudes: np.ndarray,
    eigenvectors: np.ndarray,
    n_cells: int,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
) -> np.ndarray:
    """由 Q_{q,nu} 生成双原子链的实空间位移（质量加权坐标）。"""
    if q_amplitudes.shape != (q_vals.shape[0], 2):
        raise ValueError("q_amplitudes 形状必须为 (n_q, 2)")
    if eigenvectors.shape != (q_vals.shape[0], 2, 2):
        raise ValueError("eigenvectors 形状必须为 (n_q, 2, 2)")

    n_vals = np.arange(n_cells, dtype=float)
    phases = np.exp(1j * np.outer(q_vals, n_vals * a))

    u_a = np.zeros(n_cells, dtype=complex)
    u_b = np.zeros(n_cells, dtype=complex)

    for idx, q in enumerate(q_vals):
        evecs = eigenvectors[idx]
        for mode in range(2):
            amp = q_amplitudes[idx, mode]
            factor_a = evecs[0, mode] / np.sqrt(mass_a)
            factor_b = evecs[1, mode] / np.sqrt(mass_b)
            u_a += phases[idx] * factor_a * amp / np.sqrt(n_cells)
            u_b += phases[idx] * factor_b * amp / np.sqrt(n_cells)

    return np.vstack([u_a, u_b])
