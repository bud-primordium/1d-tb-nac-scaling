"""直接带隙线性响应选择定则测试（SSH）。"""

from __future__ import annotations

import numpy as np

from src.gap_selection_rule import compute_gap_response
from src.lattice import q_grid


def test_gap_response_vanishes_for_q_nonzero() -> None:
    """对同一 k 的对角响应，q!=0 应为数值零量级。"""
    n_cells = 20
    a = 1.0
    # 选一个远离 BZ 边界的 k（k=2π*2/N）
    k = 2.0 * np.pi * 2 / (n_cells * a)
    q_vals = q_grid(n_cells, a=a)

    result = compute_gap_response(
        n_cells=n_cells,
        k=k,
        q_vals=q_vals,
        t0=1.0,
        delta_t=0.2,
        alpha=0.5,
        k_spring=1.0,
        a=a,
        mass_a=1.0,
        mass_b=1.0,
    )

    # 排除 q=0 点（第一项），其余应接近 0
    nonzero = result.gap_response[1:]
    assert float(np.max(nonzero)) < 1e-20

