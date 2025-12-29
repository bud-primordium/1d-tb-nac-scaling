"""数值 NAC 验证：解析与数值互校。"""

from __future__ import annotations

from typing import Literal, Union, Dict

import numpy as np

from .diagnostics import select_bloch_states, select_k_from_ratio
from .electron_phonon import g_monatomic, g_ssh_diatomic
from .lattice import k_grid
from .numerical_nac import align_phase, nac_finite_diff
from .phonon_1atom import displacement_from_q as displacement_from_q_mono
from .phonon_diatomic import diatomic_modes, displacement_from_q as displacement_from_q_diatomic
from .ssh_electron import (
    bloch_eigensystem,
    bloch_state as ssh_bloch_state,
    build_hamiltonian as build_hamiltonian_ssh,
    diagonalize as diagonalize_ssh,
)
from .tb_electron_1band import (
    bloch_state,
    build_hamiltonian as build_hamiltonian_tb,
    diagonalize,
    dispersion,
)


def _select_state_by_overlap(
    evals: np.ndarray,
    evecs: np.ndarray,
    psi_ref: np.ndarray,
    energy_tol: float = 1e-6,
) -> np.ndarray:
    """在退化子空间内重构与参考态最匹配的本征态。"""
    overlaps = evecs.conj().T @ psi_ref
    idx = int(np.argmax(np.abs(overlaps)))
    energy_ref = evals[idx]
    close = np.where(np.abs(evals - energy_ref) <= energy_tol)[0]
    if close.size <= 1:
        return evecs[:, idx]
    proj = evecs[:, close] @ overlaps[close]
    norm = np.linalg.norm(proj)
    if norm == 0:
        return evecs[:, idx]
    return proj / norm


def _relative_error(value_ref: complex, value_test: complex, eps: float = 1e-12) -> float:
    """计算相对误差。"""
    denom = max(np.abs(value_ref), eps)
    return float(np.abs(value_test - value_ref) / denom)


def _select_gap_k(n_cells: int, t0: float, delta_t: float, a: float) -> float:
    """在离散 k 网格上寻找最小带隙对应的 k。"""
    k_vals = k_grid(n_cells, a=a)
    gaps = []
    for k in k_vals:
        evals_k, _ = bloch_eigensystem(k, t0=t0, delta_t=delta_t, a=a)
        gaps.append(evals_k[1] - evals_k[0])
    gaps = np.array(gaps, dtype=float)
    return float(k_vals[int(np.argmin(gaps))])


def compare_analytical_numerical(
    n_cells: int,
    model_type: Literal["tb", "ssh"],
    r1: float = 1 / 10,
    r2: float = 3 / 10,
    delta: float = 1e-3,
    t0: float = 1.0,
    delta_t: float = 0.2,
    alpha: float = 0.5,
    k_spring: float = 1.0,
    a: float = 1.0,
    mass: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
    return_details: bool = False,
) -> Union[float, Dict[str, complex]]:
    """对比解析与数值 NAC，默认返回相对误差。"""
    if model_type == "tb":
        k1, k2, m1, m2 = select_bloch_states(n_cells, r1, r2, a=a)
        psi_i_ref = bloch_state(n_cells, k1, a=a)
        psi_j_ref = bloch_state(n_cells, k2, a=a)
        q_index = (m1 - m2) % n_cells
        q0 = 2.0 * np.pi * q_index / (n_cells * a)

        g_plus = g_monatomic(psi_i_ref, psi_j_ref, n_cells, q0, alpha, a=a, mass=mass)
        g_minus = g_monatomic(psi_i_ref, psi_j_ref, n_cells, -q0, alpha, a=a, mass=mass)
        g_analytical = (g_plus + g_minus) / np.sqrt(2.0)
        delta_e = dispersion(k2, t0, a=a) - dispersion(k1, t0, a=a)
        nac_analytical = g_analytical / delta_e

        q_vals = np.array([q0, -q0], dtype=float)
        q_amp = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=complex)
        u_base = displacement_from_q_mono(q_vals, q_amp, n_cells, a=a, mass=mass)
        u_base = np.real_if_close(u_base)

        h_plus = build_hamiltonian_tb(
            n_cells,
            t0=t0,
            pbc=True,
            displacements=delta * u_base,
            alpha=alpha,
        )
        h_minus = build_hamiltonian_tb(
            n_cells,
            t0=t0,
            pbc=True,
            displacements=-delta * u_base,
            alpha=alpha,
        )

        evals_plus, evecs_plus = diagonalize(h_plus)
        evals_minus, evecs_minus = diagonalize(h_minus)
        psi_j_plus = _select_state_by_overlap(evals_plus, evecs_plus, psi_j_ref)
        psi_j_minus = _select_state_by_overlap(evals_minus, evecs_minus, psi_j_ref)
        psi_j_plus = align_phase(psi_j_ref, psi_j_plus)
        psi_j_minus = align_phase(psi_j_ref, psi_j_minus)
        dpsi_j = nac_finite_diff(psi_j_plus, psi_j_minus, delta)
        nac_numerical = np.vdot(psi_i_ref, dpsi_j)

    elif model_type == "ssh":
        k_ref, _ = select_k_from_ratio(n_cells, r1, a=a)
        evals_k, evecs_k = bloch_eigensystem(k_ref, t0=t0, delta_t=delta_t, a=a)
        psi_v_ref = ssh_bloch_state(n_cells, k_ref, evecs_k[:, 0], a=a)
        psi_c_ref = ssh_bloch_state(n_cells, k_ref, evecs_k[:, 1], a=a)
        delta_e = float(evals_k[1] - evals_k[0])

        omega_q, evec_q = diatomic_modes(0.0, k_spring, mass_a, mass_b, a=a)
        mode_idx = int(np.argmax(omega_q))
        g_analytical = g_ssh_diatomic(
            psi_v_ref,
            psi_c_ref,
            n_cells,
            q=0.0,
            evec=evec_q[:, mode_idx],
            alpha=alpha,
            a=a,
            mass_a=mass_a,
            mass_b=mass_b,
        )
        nac_analytical = g_analytical / delta_e

        q_vals = np.array([0.0], dtype=float)
        q_amp = np.zeros((1, 2), dtype=complex)
        q_amp[0, mode_idx] = 1.0
        u_ab = displacement_from_q_diatomic(
            q_vals,
            q_amp,
            np.array([evec_q], dtype=float),
            n_cells,
            a=a,
            mass_a=mass_a,
            mass_b=mass_b,
        )
        u_ab = np.real_if_close(u_ab)

        h_plus = build_hamiltonian_ssh(
            n_cells,
            t0=t0,
            delta_t=delta_t,
            pbc=True,
            displacements=delta * u_ab,
            alpha=alpha,
        )
        h_minus = build_hamiltonian_ssh(
            n_cells,
            t0=t0,
            delta_t=delta_t,
            pbc=True,
            displacements=-delta * u_ab,
            alpha=alpha,
        )

        evals_plus, evecs_plus = diagonalize_ssh(h_plus)
        evals_minus, evecs_minus = diagonalize_ssh(h_minus)
        psi_c_plus = _select_state_by_overlap(evals_plus, evecs_plus, psi_c_ref)
        psi_c_minus = _select_state_by_overlap(evals_minus, evecs_minus, psi_c_ref)
        psi_c_plus = align_phase(psi_c_ref, psi_c_plus)
        psi_c_minus = align_phase(psi_c_ref, psi_c_minus)
        dpsi_c = nac_finite_diff(psi_c_plus, psi_c_minus, delta)
        nac_numerical = np.vdot(psi_v_ref, dpsi_c)

    else:
        raise ValueError("model_type 必须为 'tb' 或 'ssh'")

    rel_error = _relative_error(nac_analytical, nac_numerical)
    if return_details:
        return {
            "relative_error": rel_error,
            "nac_analytical": nac_analytical,
            "nac_numerical": nac_numerical,
        }
    return rel_error
