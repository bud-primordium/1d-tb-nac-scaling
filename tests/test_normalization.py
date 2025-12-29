"""归一化单元测试。"""

import numpy as np

from src.phonon_1atom import (
    displacement_from_q,
    kinetic_energy_from_qdot,
    kinetic_energy_from_udot,
    velocity_from_qdot,
)


def test_displacement_normalization_single_mode() -> None:
    """验证单模时 sum |u_n|^2 = 1。"""
    n_cells = 8
    a = 1.0
    mass = 1.0
    q_vals = np.array([2.0 * np.pi / (n_cells * a)])
    q_amp = np.array([1.0 + 0.0j])
    u = displacement_from_q(q_vals, q_amp, n_cells, a=a, mass=mass)
    norm = np.sum(np.abs(u) ** 2)
    assert np.isclose(norm, 1.0, atol=1e-12)


def test_kinetic_energy_consistency() -> None:
    """验证 T = 1/2 sum |Qdot|^2 与实空间动能一致。"""
    rng = np.random.default_rng(123)
    n_cells = 10
    a = 1.0
    mass = 1.0
    q_vals = 2.0 * np.pi * np.arange(n_cells) / (n_cells * a)
    qdot = rng.normal(size=n_cells) + 1j * rng.normal(size=n_cells)
    u_dot = velocity_from_qdot(q_vals, qdot, n_cells, a=a, mass=mass)
    t_q = kinetic_energy_from_qdot(qdot)
    t_u = kinetic_energy_from_udot(u_dot, mass=mass)
    assert np.isclose(t_q, t_u, atol=1e-10)
