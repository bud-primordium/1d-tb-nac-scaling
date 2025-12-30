"""直接带隙的线性响应选择定则验证（SSH 两带 + 双原子链）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .electron_phonon import g_ssh_diatomic
from .phonon_diatomic import diatomic_modes_grid
from .ssh_electron import bloch_eigensystem, bloch_state


@dataclass(frozen=True)
class GapResponseResult:
    """带隙一阶响应诊断结果。"""

    q_vals: np.ndarray
    g_vv: np.ndarray
    g_cc: np.ndarray
    gap_response: np.ndarray


def diagonal_coupling_ssh(
    n_cells: int,
    k: float,
    q_vals: np.ndarray,
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算 SSH 模型中 VBM/CBM 的对角电声耦合矩阵元。

    对于同一 k 点的 VBM 和 CBM：
        g_vv(q,nu) = <v,k|∂H/∂Q_{q,nu}|v,k>
        g_cc(q,nu) = <c,k|∂H/∂Q_{q,nu}|c,k>

    返回：
        g_vv, g_cc: 形状均为 (n_q, 2)，第二维对应声学/光学两支。
    """
    q_vals = np.asarray(q_vals, dtype=float)
    evals_k, evecs_k = bloch_eigensystem(k, t0=t0, delta_t=delta_t, a=a)
    psi_v = bloch_state(n_cells, k, evecs_k[:, 0], a=a)
    psi_c = bloch_state(n_cells, k, evecs_k[:, 1], a=a)

    _ = evals_k
    omegas, evecs_q = diatomic_modes_grid(q_vals, k_spring, mass_a, mass_b, a=a)
    _ = omegas

    g_vv = np.zeros((q_vals.shape[0], 2), dtype=complex)
    g_cc = np.zeros((q_vals.shape[0], 2), dtype=complex)

    for qi, q in enumerate(q_vals):
        for mode in range(2):
            evec = evecs_q[qi][:, mode]
            g_vv[qi, mode] = g_ssh_diatomic(
                psi_v,
                psi_v,
                n_cells,
                q=q,
                evec=evec,
                alpha=alpha,
                a=a,
                mass_a=mass_a,
                mass_b=mass_b,
            )
            g_cc[qi, mode] = g_ssh_diatomic(
                psi_c,
                psi_c,
                n_cells,
                q=q,
                evec=evec,
                alpha=alpha,
                a=a,
                mass_a=mass_a,
                mass_b=mass_b,
            )
    return g_vv, g_cc


def gap_shift_linear_coefficient(g_vv: np.ndarray, g_cc: np.ndarray) -> np.ndarray:
    """计算带隙一阶响应强度 |g_cc - g_vv|^2。

    说明：
    - 返回形状与输入一致。
    - 若直接带隙 VBM/CBM 在同一 k 点且无简并问题，
      则除 q=0 外应为数值零（动量选择定则）。
    """
    g_vv = np.asarray(g_vv)
    g_cc = np.asarray(g_cc)
    if g_vv.shape != g_cc.shape:
        raise ValueError("g_vv 与 g_cc 形状必须一致")
    return np.abs(g_cc - g_vv) ** 2


def compute_gap_response(
    n_cells: int,
    k: float,
    q_vals: np.ndarray,
    t0: float,
    delta_t: float,
    alpha: float,
    k_spring: float,
    a: float = 1.0,
    mass_a: float = 1.0,
    mass_b: float = 1.0,
) -> GapResponseResult:
    """一站式计算：返回 q 网格上的 (g_vv, g_cc, |g_cc-g_vv|^2)。"""
    g_vv, g_cc = diagonal_coupling_ssh(
        n_cells=n_cells,
        k=k,
        q_vals=q_vals,
        t0=t0,
        delta_t=delta_t,
        alpha=alpha,
        k_spring=k_spring,
        a=a,
        mass_a=mass_a,
        mass_b=mass_b,
    )
    resp = gap_shift_linear_coefficient(g_vv, g_cc)
    return GapResponseResult(
        q_vals=np.asarray(q_vals, dtype=float),
        g_vv=g_vv,
        g_cc=g_cc,
        gap_response=resp,
    )

