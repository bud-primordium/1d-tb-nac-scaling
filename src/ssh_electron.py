"""SSH 两带电子模型。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .lattice import pbc_index


def build_hamiltonian(
    n_cells: int,
    t0: float = 1.0,
    delta_t: float = 0.2,
    onsite: Optional[np.ndarray] = None,
    pbc: bool = True,
) -> np.ndarray:
    """构造 SSH 两带哈密顿量（A1,B1,A2,B2 顺序）。"""
    size = 2 * n_cells
    if onsite is None:
        onsite = np.zeros(size, dtype=float)
    if onsite.shape[0] != size:
        raise ValueError("onsite 长度必须为 2*n_cells")

    h = np.zeros((size, size), dtype=complex)
    np.fill_diagonal(h, onsite)

    v = t0 + delta_t
    w = t0 - delta_t

    for n in range(n_cells):
        a_idx = 2 * n
        b_idx = 2 * n + 1
        h[a_idx, b_idx] += v
        h[b_idx, a_idx] += v

        n_next = n + 1
        if n_next >= n_cells:
            if not pbc:
                continue
            n_next = pbc_index(n_next, n_cells)
        a_next_idx = 2 * n_next
        h[b_idx, a_next_idx] += w
        h[a_next_idx, b_idx] += w

    return h


def diagonalize(h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """对角化哈密顿量，返回本征值与本征矢（列向量）。"""
    evals, evecs = np.linalg.eigh(h)
    return evals, evecs


def band_energies(k_vals: np.ndarray, t0: float = 1.0, delta_t: float = 0.2, a: float = 1.0) -> np.ndarray:
    """解析能带：E_±(k)。"""
    v = t0 + delta_t
    w = t0 - delta_t
    energies = np.sqrt(v ** 2 + w ** 2 + 2.0 * v * w * np.cos(k_vals * a))
    return np.vstack([-energies, energies])


def bloch_hamiltonian(k: float, t0: float = 1.0, delta_t: float = 0.2, a: float = 1.0) -> np.ndarray:
    """SSH 的 2x2 Bloch 哈密顿量。"""
    v = t0 + delta_t
    w = t0 - delta_t
    offdiag = v + w * np.exp(-1j * k * a)
    h = np.array([[0.0, offdiag], [np.conj(offdiag), 0.0]], dtype=complex)
    return h


def bloch_eigensystem(
    k: float,
    t0: float = 1.0,
    delta_t: float = 0.2,
    a: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 2x2 Bloch 哈密顿量的本征值与本征矢。"""
    h = bloch_hamiltonian(k, t0=t0, delta_t=delta_t, a=a)
    evals, evecs = np.linalg.eigh(h)
    return evals, evecs


def bloch_state(n_cells: int, k: float, sublattice_vec: np.ndarray, a: float = 1.0) -> np.ndarray:
    """由 2 分量子格矢构造 real-space Bloch 态。"""
    phases = np.exp(1j * k * np.arange(n_cells) * a) / np.sqrt(n_cells)
    psi = np.zeros(2 * n_cells, dtype=complex)
    psi[0::2] = sublattice_vec[0] * phases
    psi[1::2] = sublattice_vec[1] * phases
    return psi


def build_square_well(
    n_cells: int,
    center: int,
    width: int,
    depth: float,
) -> np.ndarray:
    """构造 SSH 模型的方势阱 on-site 能量（作用于 A/B）。"""
    size = 2 * n_cells
    onsite = np.zeros(size, dtype=float)
    half = width // 2
    for offset in range(-half, half + 1):
        idx = pbc_index(center + offset, n_cells)
        onsite[2 * idx] = depth
        onsite[2 * idx + 1] = depth
    return onsite
