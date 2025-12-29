# 1D Tight-Binding NAC 尺寸效应验证

本项目用于验证非绝热耦合（NAC）的尺寸标度律，包含：
- 简化版：单带单原子链（Case 1′/2/3）
- 完整版：SSH 两带 + 双原子链（Case 1/2/3 与折叠声子验证）

## 目录结构

```
1229_1dTB/
├── src/                 # 核心模块
├── notebooks/           # 实验 notebook
├── tests/               # 单元测试
├── results/             # 输出图表与数据
├── .gitignore
└── README.md
```

## 运行说明

1. 安装依赖（示例）：
   - numpy
   - scipy
   - matplotlib
   - pytest

2. 运行测试：
```
pytest -q
```

3. 运行脚本（推荐）：
```
python scripts/run_simple.py
python scripts/run_ssh.py
```
图表统一输出到 `results/figures/`。

4. 运行 notebook：
在 `notebooks/` 中逐个执行即可，图表统一输出到 `results/figures/`。

## 约定

- 所有注释与 docstring 使用中文
- 归一化与坐标约定必须与执行计划一致
- 图表输出统一保存为 PDF + PNG
