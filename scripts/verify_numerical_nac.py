"""验证数值 NAC 计算 - 解析方法 vs 数值方法对比。

此脚本生成附录图表，展示两种计算方法的一致性。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt

from src.experiments_numerical_check import compare_analytical_numerical


# =============================================================================
# 样式设置
# =============================================================================

COLORS = {
    'analytical': '#1f77b4',  # 蓝色
    'numerical': '#ff7f0e',   # 橙色
    'error': '#d62728',       # 红色
}


def setup_style():
    """设置图表样式。"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'axes.unicode_minus': False,
        'xtick.major.size': 4,
        'xtick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.major.size': 4,
        'ytick.labelsize': 10,
        'ytick.direction': 'in',
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# 主验证函数
# =============================================================================

def verify_numerical_methods():
    """对比解析与数值方法，生成验证图表。"""
    setup_style()

    # 参数设置
    n_vals = [20, 40, 60, 80, 120, 160, 240]

    # 数据收集
    tb_analytical = []
    tb_numerical = []
    tb_errors = []

    ssh_analytical = []
    ssh_numerical = []
    ssh_errors = []

    print("Computing TB model...")
    for n in n_vals:
        result = compare_analytical_numerical(n, "tb", return_details=True)
        tb_analytical.append(np.abs(result['nac_analytical']))
        tb_numerical.append(np.abs(result['nac_numerical']))
        tb_errors.append(result['relative_error'])
        print(f"  N={n:3d}: error={result['relative_error']:.2e}")

    print("\nComputing SSH model...")
    for n in n_vals:
        result = compare_analytical_numerical(n, "ssh", return_details=True)
        ssh_analytical.append(np.abs(result['nac_analytical']))
        ssh_numerical.append(np.abs(result['nac_numerical']))
        ssh_errors.append(result['relative_error'])
        print(f"  N={n:3d}: error={result['relative_error']:.2e}")

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    # (a) TB NAC 值对比
    ax = axes[0, 0]
    ax.plot(n_vals, tb_analytical, 'o-', color=COLORS['analytical'],
            label='Analytical', markersize=5)
    ax.plot(n_vals, tb_numerical, 's--', color=COLORS['numerical'],
            label='Numerical', markersize=4, alpha=0.7)
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$|d_{ij}|$')
    ax.set_title('(a) TB: NAC magnitude', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle=':')

    # (b) TB 相对误差
    ax = axes[0, 1]
    ax.semilogy(n_vals, tb_errors, 'o-', color=COLORS['error'], markersize=5)
    ax.axhline(y=1e-4, linestyle='--', color='gray', linewidth=1,
               alpha=0.7, label=r'Target $10^{-4}$')
    ax.set_xlabel('$N$')
    ax.set_ylabel('Relative error')
    ax.set_title('(b) TB: Method agreement', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle=':')

    # (c) SSH NAC 值对比
    ax = axes[1, 0]
    ax.plot(n_vals, ssh_analytical, 'o-', color=COLORS['analytical'],
            label='Analytical', markersize=5)
    ax.plot(n_vals, ssh_numerical, 's--', color=COLORS['numerical'],
            label='Numerical', markersize=4, alpha=0.7)
    ax.set_xlabel('$N$')
    ax.set_ylabel(r'$|d_{ij}|$')
    ax.set_title('(c) SSH: NAC magnitude', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle=':')

    # (d) SSH 相对误差
    ax = axes[1, 1]
    ax.semilogy(n_vals, ssh_errors, 'o-', color=COLORS['error'], markersize=5)
    ax.axhline(y=1e-4, linestyle='--', color='gray', linewidth=1,
               alpha=0.7, label=r'Target $10^{-4}$')
    ax.set_xlabel('$N$')
    ax.set_ylabel('Relative error')
    ax.set_title('(d) SSH: Method agreement', fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    # 保存
    output_dir = ROOT / "results" / "publication_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fig_appendix_numerical_verification.png"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"\n✓ Verification figure saved to: {output_path}")

    # 汇总统计
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    print(f"TB model:")
    print(f"  Max error: {max(tb_errors):.2e}")
    print(f"  Mean error: {np.mean(tb_errors):.2e}")
    print(f"  All < 1e-4: {all(e < 1e-4 for e in tb_errors)}")
    print(f"\nSSH model:")
    print(f"  Max error: {max(ssh_errors):.2e}")
    print(f"  Mean error: {np.mean(ssh_errors):.2e}")
    print(f"  All < 1e-4: {all(e < 1e-4 for e in ssh_errors)}")
    print("="*60)


if __name__ == "__main__":
    verify_numerical_methods()
