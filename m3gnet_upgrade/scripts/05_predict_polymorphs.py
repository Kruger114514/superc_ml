"""
05_predict_polymorphs.py — 多形态对比 (报告 v2 的核心 figure)
===========================================================

证明 v2 (M3GNet) 能区分同组分不同结构, v1 (XGBoost) 不能。

对比的多形态对:
  1. Diamond vs Graphite (C, 化学式都是 C)
  2. α-Fe (BCC) vs γ-Fe (FCC)
  3. (可选) H3S 不同压力下的相 (如果能拿到 cif)

输出:
  - results/polymorph_comparison.csv (数值表)
  - results/figures/polymorph_comparison.png (条形图)

运行: python scripts/05_predict_polymorphs.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 把 scripts 加入 path 以复用 predictor
sys.path.insert(0, str(ROOT))


def make_diamond():
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(3.567),
                                     ["C"], [[0, 0, 0]])


def make_graphite():
    a, c = 2.461, 6.708
    lattice = Lattice.hexagonal(a, c)
    coords = [[0, 0, 0], [0, 0, 0.5], [1/3, 2/3, 0], [2/3, 1/3, 0.5]]
    return Structure(lattice, ["C"] * 4, coords)


def make_bcc_iron():
    return Structure.from_spacegroup("Im-3m", Lattice.cubic(2.866),
                                     ["Fe"], [[0, 0, 0]])


def make_fcc_iron():
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(3.520),
                                     ["Fe"], [[0, 0, 0]])


POLYMORPHS = [
    {"name": "Diamond",  "structure": make_diamond,  "formula": "C",  "group": "Carbon"},
    {"name": "Graphite", "structure": make_graphite, "formula": "C",  "group": "Carbon"},
    {"name": "α-Fe",     "structure": make_bcc_iron, "formula": "Fe", "group": "Iron"},
    {"name": "γ-Fe",     "structure": make_fcc_iron, "formula": "Fe", "group": "Iron"},
]


def predict_v2(records):
    """用我们训练好的 M3GNet Tc 模型预测"""
    from m3gnet_predictor import M3GNetTcPredictor
    predictor = M3GNetTcPredictor(
        ckpt_path=ROOT / "models" / "m3gnet_tc.pt",
        config_path=ROOT / "configs" / "train_config.yaml",
    )
    return [predictor.predict_structure(r["structure"]) for r in records]


def predict_v1(records):
    """用 v1 (XGBoost) 预测 — 注意 v1 只看化学式"""
    # 切换到 v1 项目目录加载
    v1_root = ROOT.parent  # superc_ml/
    sys.path.insert(0, str(v1_root))
    try:
        from predict import TcPredictor
        v1_pred = TcPredictor(
            model_path=str(v1_root / "models" / "tc_predictor.pkl"),
            train_data_path=str(v1_root / "data" / "raw" / "unique_m.csv"),
        )
        return [v1_pred.predict(r["formula"])["tc_pred"] for r in records]
    except Exception as e:
        print(f"  ⚠ v1 模型加载失败 ({e}), v1 列将为 NaN")
        return [np.nan] * len(records)
    finally:
        sys.path.remove(str(v1_root))


def main():
    print("=" * 70)
    print("  Polymorph Comparison: v2 (M3GNet) vs v1 (XGBoost)")
    print("=" * 70)

    # 准备多形态
    print("\n[1/3] 准备多形态结构...")
    records = []
    for poly in POLYMORPHS:
        struct = poly["structure"]()
        records.append({
            "name": poly["name"],
            "formula": poly["formula"],
            "group": poly["group"],
            "structure": struct,
        })
        print(f"      {poly['name']:<15} formula={poly['formula']:<5} "
              f"n_atoms={len(struct)}, V={struct.volume:.2f} Å³")

    # v2 预测
    print("\n[2/3] v2 (M3GNet) 预测...")
    v2_preds = predict_v2(records)

    # v1 预测
    print("\n[3/3] v1 (XGBoost) 预测...")
    v1_preds = predict_v1(records)

    # 整理结果
    df = pd.DataFrame({
        "Polymorph": [r["name"] for r in records],
        "Formula":   [r["formula"] for r in records],
        "Group":     [r["group"] for r in records],
        "v1_pred_tc": v1_preds,
        "v2_pred_tc": v2_preds,
    })

    print()
    print(df.to_string(index=False))

    # 关键观察 + 保存
    print("\n关键观察 (按 Group 看 v1 vs v2 的方差):")
    for group, grp in df.groupby("Group"):
        if len(grp) < 2:
            continue
        v1_var = grp["v1_pred_tc"].std()
        v2_var = grp["v2_pred_tc"].std()
        print(f"  {group}: v1 std = {v1_var:.3f} K, v2 std = {v2_var:.3f} K")
        if v1_var < 0.01:
            print(f"        → v1 给出相同预测 (只看化学式)")
        if v2_var > 0.01:
            print(f"        → v2 区分多形态 (看到结构)")

    # 保存表
    csv_path = ROOT / "results" / "polymorph_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 数据表已保存到 {csv_path}")

    # 画图
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df["v1_pred_tc"], width, label="v1 (XGBoost, composition-only)",
           color="#3b82f6", alpha=0.8, edgecolor="black")
    ax.bar(x + width/2, df["v2_pred_tc"], width, label="v2 (M3GNet, structure-aware)",
           color="#dc2626", alpha=0.8, edgecolor="black", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Polymorph"], rotation=15)
    ax.set_ylabel(r"Predicted $T_c$ (K)", fontsize=12)
    ax.set_title("Polymorph predictions: v1 cannot distinguish, v2 can", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    # 标注同 group 的连线
    for group in df["Group"].unique():
        grp_x = x[df["Group"] == group]
        if len(grp_x) == 2:
            ymax = max(df.loc[df["Group"] == group, "v2_pred_tc"].max(),
                       df.loc[df["Group"] == group, "v1_pred_tc"].max()) + 1
            ax.annotate(f"Same formula ({df[df['Group'] == group].iloc[0]['Formula']})",
                        xy=((grp_x[0] + grp_x[1]) / 2, ymax),
                        ha="center", fontsize=9, style="italic", color="gray")
    plt.tight_layout()
    out_path = FIG_DIR / "polymorph_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 图已保存到 {out_path}")

    print()
    print("=" * 70)
    print("  完成. 这张图可以直接放进报告 v2 的 §6.4")
    print("=" * 70)


if __name__ == "__main__":
    main()
