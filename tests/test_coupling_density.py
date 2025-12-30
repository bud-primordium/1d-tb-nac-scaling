"""耦合密度的单元测试。"""

from __future__ import annotations

import numpy as np

from src.coupling_density import coupling_density_realspace


def test_coupling_density_sum_rule() -> None:
    """sum_n D(n) 应返回总矩阵元 <i|dH|j>。"""
    rng = np.random.default_rng(0)
    n_dim = 8
    psi_i = rng.normal(size=n_dim) + 1j * rng.normal(size=n_dim)
    psi_j = rng.normal(size=n_dim) + 1j * rng.normal(size=n_dim)
    psi_i = psi_i / np.linalg.norm(psi_i)
    psi_j = psi_j / np.linalg.norm(psi_j)

    dh = rng.normal(size=(n_dim, n_dim)) + 1j * rng.normal(size=(n_dim, n_dim))
    d = coupling_density_realspace(psi_i, psi_j, dh)

    total_from_density = np.sum(d)
    total_direct = np.vdot(psi_i, dh @ psi_j)
    assert np.allclose(total_from_density, total_direct, atol=1e-12)

