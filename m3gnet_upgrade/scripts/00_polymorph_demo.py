"""
00_polymorph_demo.py — 最小演示: 结构敏感性证明
================================================

目的: 不需要训练, 不需要 GPU, 用 matgl 的预训练 formation energy 模型
对比同组分不同结构的 (石墨 vs 金刚石) 能量预测。

预期: 两者数值显著不同 -> 证明 "结构进入模型后是可区分的"
     -> 老师提的 "石墨和金刚石化学式相同, 温度是否一样" 问题
     -> 现在有了一个 ML 系统能区分它们

运行: python scripts/00_polymorph_demo.py
预计耗时: 1-2 分钟 (第一次跑会下载预训练权重)
"""
from __future__ import annotations

import warnings
warnings.simplefilter("ignore")

import sys
from pathlib import Path

import matgl
import torch
from pymatgen.core import Lattice, Structure


# =============================================================================
# 1. 准备多形态结构
# =============================================================================

def make_diamond() -> Structure:
    """金刚石: 面心立方 (Fd-3m), 晶格常数 a = 3.567 Å"""
    a = 3.567
    return Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(a), ["C"], [[0, 0, 0]]
    )


def make_graphite() -> Structure:
    """石墨: 六方 (P6_3/mmc), a = 2.461 Å, c = 6.708 Å"""
    a, c = 2.461, 6.708
    lattice = Lattice.hexagonal(a, c)
    species = ["C", "C", "C", "C"]
    coords = [
        [0, 0, 0],
        [0, 0, 0.5],
        [1/3, 2/3, 0],
        [2/3, 1/3, 0.5],
    ]
    return Structure(lattice, species, coords)


def make_alpha_iron() -> Structure:
    """α-Fe (BCC, Im-3m), a = 2.866 Å — 用于额外对比"""
    a = 2.866
    return Structure.from_spacegroup(
        "Im-3m", Lattice.cubic(a), ["Fe"], [[0, 0, 0]]
    )


def make_gamma_iron() -> Structure:
    """γ-Fe (FCC, Fm-3m), a = 3.520 Å"""
    a = 3.520
    return Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(a), ["Fe"], [[0, 0, 0]]
    )


# =============================================================================
# 2. 加载预训练模型 + 预测
# =============================================================================

def main():
    print("=" * 70)
    print("  M3GNet Polymorph Demo")
    print("  证明: 结构进入模型后, 同组分不同结构有不同预测")
    print("=" * 70)

    # 加载预训练 M3GNet formation energy 模型
    # 这一步会从 HuggingFace 下载权重 (约 10MB), 第一次跑需要联网
    print("\n[1/3] 加载预训练 M3GNet 模型 (M3GNet-MP-2018.6.1-Eform)...")
    model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
    print(f"      ✓ 模型加载完成, 设备: {next(model.parameters()).device}")

    # 准备所有结构
    print("\n[2/3] 准备多形态结构...")
    polymorphs = {
        "Diamond (cubic, Fd-3m)":      make_diamond(),
        "Graphite (hexagonal, P63/mmc)": make_graphite(),
        "α-Fe (BCC)":                  make_alpha_iron(),
        "γ-Fe (FCC)":                  make_gamma_iron(),
    }
    for name, struct in polymorphs.items():
        print(f"      {name}: {struct.composition.reduced_formula}, "
              f"{len(struct)} atoms, V={struct.volume:.2f} Å³")

    # 预测 formation energy
    print("\n[3/3] 预测 formation energy...")
    print()
    print(f"  {'Structure':<35} {'Composition':<10} {'E_form (eV/atom)':>20}")
    print("  " + "-" * 67)

    results = []
    for name, struct in polymorphs.items():
        eform = model.predict_structure(struct)
        eform_val = float(eform)
        results.append((name, struct.composition.reduced_formula, eform_val))
        print(f"  {name:<35} {struct.composition.reduced_formula:<10} {eform_val:>20.4f}")

    # 关键比较
    print()
    print("=" * 70)
    print("  关键观察 (对比同组分不同结构)")
    print("=" * 70)

    diamond_e = next(e for n, _, e in results if "Diamond" in n)
    graphite_e = next(e for n, _, e in results if "Graphite" in n)
    fe_alpha_e = next(e for n, _, e in results if "α-Fe" in n)
    fe_gamma_e = next(e for n, _, e in results if "γ-Fe" in n)

    print(f"\n  Diamond  E_form = {diamond_e:+.4f} eV/atom")
    print(f"  Graphite E_form = {graphite_e:+.4f} eV/atom")
    print(f"  ΔE (diamond - graphite) = {diamond_e - graphite_e:+.4f} eV/atom")
    print(f"  (实验值约 +0.02 eV/atom: 石墨更稳定)")

    print(f"\n  α-Fe E_form = {fe_alpha_e:+.4f} eV/atom")
    print(f"  γ-Fe E_form = {fe_gamma_e:+.4f} eV/atom")
    print(f"  ΔE (γ - α)  = {fe_gamma_e - fe_alpha_e:+.4f} eV/atom")
    print(f"  (实验上 α-Fe 在室温更稳定)")

    print()
    print("=" * 70)
    print("  结论:")
    print("  - 化学式相同, 结构不同 -> M3GNet 给出不同预测")
    print("  - 这正是 v1 的 Magpie + XGBoost 无法做到的")
    print("  - 下一步: 用 3DSC 数据集 fine-tune M3GNet 预测 Tc")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 出错: {e}\n", file=sys.stderr)
        print("常见问题排查:", file=sys.stderr)
        print("  1. 是否装了 matgl?  pip install matgl", file=sys.stderr)
        print("  2. 第一次跑需要联网下载预训练权重", file=sys.stderr)
        print("  3. matgl 1.x 和 2.x API 不同, 我们用 1.x", file=sys.stderr)
        raise
