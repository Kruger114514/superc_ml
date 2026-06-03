"""
06_compare_with_v1.py — v1 (XGBoost) vs v2 (M3GNet) 完整对比
==========================================================

在同样的测试集上跑两个模型, 报告:
  - 整体: R², MAE
  - 按家族: MAE
  - 关键化合物 (LaH10, La3Ni2O7 等) 的具体预测

输出:
  - results/v1_vs_v2_comparison.csv
  - results/figures/v1_vs_v2_mae_by_family.png

运行: python scripts/06_compare_with_v1.py
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_v1_predictions_on_v2_testset():
    """
    用 v1 (XGBoost) 在 v2 的测试集上预测.

    注意: v1 模型只看化学式. 这里我们对 v2 测试集中的每个化学式调用 v1.
    """
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parent))  # superc_ml/

    try:
        from predict import TcPredictor
    except ImportError:
        print(f"  ⚠ 无法 import v1 的 predict.py")
        return None

    v1_pred = TcPredictor(
        model_path=str(ROOT.parent / "models" / "tc_predictor.pkl"),
        train_data_path=str(ROOT.parent / "data" / "raw" / "unique_m.csv"),
    )

    # 加载 v2 测试集
    with open(ROOT / "data" / "processed" / "test.pkl", "rb") as f:
        test_records = pickle.load(f)

    results = []
    for r in test_records:
        try:
            v1_out = v1_pred.predict(r["formula"])
            v1_pred_tc = v1_out["tc_pred"]
        except Exception:
            v1_pred_tc = np.nan
        results.append({
            "formula": r["formula"],
            "family": r["family"],
            "true_tc": r["tc"],
            "v1_pred_tc": v1_pred_tc,
        })
    return pd.DataFrame(results)


def load_v2_predictions():
    """直接读 04_evaluate.py 的输出"""
    path = ROOT / "results" / "v2_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}, 请先跑 04_evaluate.py")
    return pd.read_csv(path)


def main():
    print("=" * 70)
    print("  v1 (XGBoost) vs v2 (M3GNet) 对比")
    print("=" * 70)

    print("\n[1/3] 加载 v2 预测结果...")
    v2_df = load_v2_predictions()
    print(f"      v2 测试集大小: {len(v2_df)}")

    print("[2/3] 在 v2 测试集上跑 v1 预测...")
    v1_df = load_v1_predictions_on_v2_testset()
    if v1_df is None or v1_df["v1_pred_tc"].isna().all():
        print("      ⚠ v1 预测失败, 仅报告 v2")
        return

    # 合并 (按 formula)
    merged = pd.merge(
        v1_df[["formula", "v1_pred_tc"]],
        v2_df[["formula", "family", "true_tc", "pred_tc"]],
        on="formula", how="inner",
    )
    merged = merged.rename(columns={"pred_tc": "v2_pred_tc"})
    merged = merged.dropna(subset=["v1_pred_tc", "v2_pred_tc"])
    print(f"      合并后 {len(merged)} 条")

    # 整体指标
    print("\n[3/3] 计算指标...")
    for model in ["v1", "v2"]:
        pred_col = f"{model}_pred_tc"
        r2 = r2_score(merged["true_tc"], merged[pred_col])
        mae = mean_absolute_error(merged["true_tc"], merged[pred_col])
        print(f"      {model}: R² = {r2:.3f}, MAE = {mae:.2f} K")

    # 按家族 MAE
    print("\n  按家族 MAE:")
    family_table = []
    for family, grp in merged.groupby("family"):
        v1_mae = mean_absolute_error(grp["true_tc"], grp["v1_pred_tc"])
        v2_mae = mean_absolute_error(grp["true_tc"], grp["v2_pred_tc"])
        improvement = (v1_mae - v2_mae) / v1_mae * 100 if v1_mae > 0 else 0
        family_table.append({
            "family": family, "n": len(grp),
            "v1_mae": v1_mae, "v2_mae": v2_mae,
            "improvement_pct": improvement,
        })
        print(f"    {family:<15} n={len(grp):4d}  "
              f"v1 MAE = {v1_mae:6.2f} K  v2 MAE = {v2_mae:6.2f} K  "
              f"({improvement:+.1f}%)")

    ft_df = pd.DataFrame(family_table)
    ft_df.to_csv(ROOT / "results" / "v1_vs_v2_by_family.csv", index=False)
    merged.to_csv(ROOT / "results" / "v1_vs_v2_predictions.csv", index=False)

    # 画图: 并排柱状图
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ft_df))
    width = 0.35
    ax.bar(x - width/2, ft_df["v1_mae"], width, label="v1 (XGBoost)",
           color="#3b82f6", alpha=0.8, edgecolor="black")
    ax.bar(x + width/2, ft_df["v2_mae"], width, label="v2 (M3GNet)",
           color="#dc2626", alpha=0.8, edgecolor="black", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(ft_df["family"], rotation=15)
    ax.set_ylabel("MAE (K)", fontsize=12)
    ax.set_title("Per-family MAE: v1 vs v2", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 数字标注
    for i, row in ft_df.iterrows():
        ax.text(i - width/2, row["v1_mae"] + 1, f"{row['v1_mae']:.1f}",
                ha="center", fontsize=9)
        ax.text(i + width/2, row["v2_mae"] + 1, f"{row['v2_mae']:.1f}",
                ha="center", fontsize=9)

    plt.tight_layout()
    out = FIG_DIR / "v1_vs_v2_mae_by_family.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ 图已保存到 {out}")
    print(f"  ✓ 数据表: {ROOT / 'results' / 'v1_vs_v2_by_family.csv'}")

    print()
    print("=" * 70)
    print("  对比完成. 这张图直接放进报告 v2 的 §6.4")
    print("=" * 70)


if __name__ == "__main__":
    main()
