"""
plot_v2_clean_figures.py — 生成 v2 测试集评估的 4 张报告 figure
==================================================================

输入: results/v2_test_predictions.csv  (04_evaluate.py 生成的)
输出: results/figures/
  ├── v2_pred_vs_exp_clean.png      ← 修正后的散点图 (核心 figure 1)
  ├── v2_mae_by_family.png          ← 按家族 MAE 柱状图 (核心 figure 2)
  ├── v2_error_distribution.png     ← 误差分布直方图 (辅助)
  └── v2_per_family_comparison.png  ← 单家族散点子图 (辅助)

运行: python plot_v2_clean_figures.py
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# ============= 配置 =============
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "v2_test_predictions.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTLIER_THRESHOLD = 100.0  # abs_error > 100 K 视为 OOD outlier

# 配色 (按 Tc 范围排序, 由低到高)
FAMILY_COLORS = {
    "Hydride":    "#1f77b4",   # 蓝
    "A15":        "#2ca02c",   # 绿
    "BCS-Other":  "#7f7f7f",   # 灰
    "MgB2-like":  "#ff7f0e",   # 橙
    "Iron-based": "#9467bd",   # 紫
    "Nickelate":  "#17becf",   # 青
    "Cuprate":    "#d62728",   # 红
}

# 家族在图上的展示顺序 (按典型 Tc 由低到高)
FAMILY_ORDER = ["Hydride", "BCS-Other", "A15", "MgB2-like",
                "Nickelate", "Iron-based", "Cuprate"]

# v1 baseline 对比
V1_MAE = 5.48
V1_R2 = 0.918

print("=" * 70)
print("  v2 评估 figure 生成")
print("=" * 70)

# ============= 加载数据 =============
df = pd.read_csv(CSV_PATH)
df_outliers = df[df["abs_error"] >= OUTLIER_THRESHOLD].copy()
df_clean = df[df["abs_error"] < OUTLIER_THRESHOLD].copy()

print(f"  总样本: {len(df)}")
print(f"  剔除 OOD outlier: {len(df_outliers)} ({len(df_outliers)/len(df)*100:.2f}%)")
print(f"  清洁样本: {len(df_clean)}")
print()

# ============= 整体指标 =============
overall_r2 = r2_score(df_clean["true_tc"], df_clean["pred_tc"])
overall_mae = mean_absolute_error(df_clean["true_tc"], df_clean["pred_tc"])
print(f"  Overall R²:  {overall_r2:.3f}")
print(f"  Overall MAE: {overall_mae:.2f} K")
print()

# ============= 按家族指标 =============
family_stats = []
for family in FAMILY_ORDER:
    grp = df_clean[df_clean["family"] == family]
    if len(grp) == 0:
        continue
    n = len(grp)
    mae = mean_absolute_error(grp["true_tc"], grp["pred_tc"])
    r2 = r2_score(grp["true_tc"], grp["pred_tc"]) if n > 1 else float("nan")
    family_stats.append({"family": family, "n": n, "mae": mae, "r2": r2})
family_df = pd.DataFrame(family_stats)

# ============================================================================
# Figure 1: Pred vs Exp 散点图 (修正后) - 报告核心 figure
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 8))

for family in FAMILY_ORDER:
    grp = df_clean[df_clean["family"] == family]
    if len(grp) == 0:
        continue
    color = FAMILY_COLORS.get(family, "gray")
    ax.scatter(grp["true_tc"], grp["pred_tc"],
               c=color, label=f"{family} (n={len(grp)})",
               alpha=0.7, s=50, edgecolor="black", linewidth=0.4)

# 对角线
max_tc = max(df_clean["true_tc"].max(), df_clean["pred_tc"].max()) * 1.05
ax.plot([0, max_tc], [0, max_tc], "--", color="black", lw=1.2,
        label="Perfect prediction", zorder=1)

# ±30% 带 (用 fill_between 表示)
xs = np.linspace(0, max_tc, 100)
ax.fill_between(xs, xs * 0.7, xs * 1.3, color="gray", alpha=0.08,
                label="±30% band")

ax.set_xlim(-3, max_tc)
ax.set_ylim(-3, max_tc)
ax.set_aspect("equal")
ax.set_xlabel(r"Experimental $T_c$ (K)", fontsize=13)
ax.set_ylabel(r"Predicted $T_c$ (K)", fontsize=13)
ax.set_title(
    f"v2 (M3GNet) — Predicted vs Experimental $T_c$\n"
    f"n={len(df_clean)} (1 OOD outlier removed), MAE = {overall_mae:.2f} K, R² = {overall_r2:.3f}",
    fontsize=12)
ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax.grid(alpha=0.3)

plt.tight_layout()
out1 = OUT_DIR / "v2_pred_vs_exp_clean.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {out1.name}")

# ============================================================================
# Figure 2: 按家族 MAE 柱状图 - 报告核心 figure
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

family_df_sorted = family_df.sort_values("mae")
families = family_df_sorted["family"].tolist()
maes = family_df_sorted["mae"].tolist()
ns = family_df_sorted["n"].tolist()
colors = [FAMILY_COLORS.get(f, "gray") for f in families]

x_pos = np.arange(len(families))
bars = ax.bar(x_pos, maes, color=colors, edgecolor="black", linewidth=0.6)

# 在柱顶标 MAE 数字
for i, (bar, mae, n) in enumerate(zip(bars, maes, ns)):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
            f"{mae:.2f} K\n(n={n})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

# v1 baseline 横线
ax.axhline(V1_MAE, color="black", linestyle="--", lw=1.5,
           label=f"v1 baseline (MAE = {V1_MAE} K)", zorder=10)

# v2 整体 MAE
ax.axhline(overall_mae, color="red", linestyle=":", lw=1.5,
           label=f"v2 overall (MAE = {overall_mae:.2f} K)", zorder=10)

ax.set_xticks(x_pos)
ax.set_xticklabels(families, rotation=15, ha="right")
ax.set_ylabel("Test MAE (K)", fontsize=12)
ax.set_title(
    f"v2 Per-Family Performance vs v1 Baseline\n"
    f"4/7 families achieve MAE < 4 K covering {sum(n for f, n in zip(families, ns) if maes[families.index(f)] < 4)}/{sum(ns)} samples",
    fontsize=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(maes) * 1.3)

plt.tight_layout()
out2 = OUT_DIR / "v2_mae_by_family.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {out2.name}")

# ============================================================================
# Figure 3: 误差分布直方图 (辅助 figure)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: 整体误差分布
ax = axes[0]
errors_clean = df_clean["error"].values  # pred - true
ax.hist(errors_clean, bins=50, color="steelblue", edgecolor="black",
        linewidth=0.4, alpha=0.85)
ax.axvline(0, color="black", linestyle="-", lw=1, label="Zero error")
ax.axvline(np.mean(errors_clean), color="red", linestyle="--", lw=1.5,
           label=f"Mean = {np.mean(errors_clean):.2f} K")
ax.set_xlabel("Prediction Error (K): pred - true", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Error distribution (n={len(df_clean)})\n"
             f"Median |error| = {np.median(np.abs(errors_clean)):.2f} K, "
             f"Std = {np.std(errors_clean):.2f} K",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Right: log scale 看尾部
ax = axes[1]
abs_errors = np.abs(errors_clean)
ax.hist(abs_errors, bins=np.logspace(-2, 2.5, 50), color="darkorange",
        edgecolor="black", linewidth=0.4, alpha=0.85)
ax.axvline(overall_mae, color="red", linestyle="--", lw=1.5,
           label=f"Overall MAE = {overall_mae:.2f} K")
ax.set_xscale("log")
ax.set_xlabel("Absolute Error (K) [log scale]", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Absolute error distribution (log x)\n"
             f"75% within {np.percentile(abs_errors, 75):.2f} K, "
             f"90% within {np.percentile(abs_errors, 90):.2f} K",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
out3 = OUT_DIR / "v2_error_distribution.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {out3.name}")

# ============================================================================
# Figure 4: 各家族单独散点子图 (辅助 figure)
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, family in enumerate(FAMILY_ORDER):
    ax = axes[i]
    grp = df_clean[df_clean["family"] == family]
    if len(grp) == 0:
        ax.axis("off")
        continue
    color = FAMILY_COLORS.get(family, "gray")
    ax.scatter(grp["true_tc"], grp["pred_tc"],
               c=color, alpha=0.7, s=40,
               edgecolor="black", linewidth=0.4)
    max_t = max(grp["true_tc"].max(), grp["pred_tc"].max()) * 1.1
    min_t = min(grp["true_tc"].min(), grp["pred_tc"].min()) - 1
    ax.plot([min_t, max_t], [min_t, max_t], "--", color="black", lw=0.8)
    ax.set_xlim(min_t, max_t)
    ax.set_ylim(min_t, max_t)
    ax.set_aspect("equal")
    n = len(grp)
    mae = mean_absolute_error(grp["true_tc"], grp["pred_tc"])
    r2 = r2_score(grp["true_tc"], grp["pred_tc"]) if n > 1 else float("nan")
    r2_str = f"R²={r2:.2f}" if not np.isnan(r2) else "R²=N/A"
    ax.set_title(f"{family} (n={n})\nMAE={mae:.2f}K, {r2_str}",
                 fontsize=10)
    ax.set_xlabel(r"Exp $T_c$ (K)", fontsize=9)
    ax.set_ylabel(r"Pred $T_c$ (K)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

# 隐藏第 8 个 (我们只有 7 个家族)
for j in range(len(FAMILY_ORDER), len(axes)):
    axes[j].axis("off")

plt.suptitle("v2 — Per-Family Predicted vs Experimental $T_c$", fontsize=13, y=1.00)
plt.tight_layout()
out4 = OUT_DIR / "v2_per_family_comparison.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ {out4.name}")

# ============================================================================
# 保存指标 csv (报告里画表用)
# ============================================================================
family_df["mae_v1"] = V1_MAE  # 用同一个 v1 总 MAE 做参考
family_df["v1_improvement"] = (V1_MAE - family_df["mae"]) / V1_MAE * 100
family_df.columns = ["Family", "n", "MAE_v2 (K)", "R²_v2", "MAE_v1_overall (K)", "v2 vs v1 (%)"]
csv_out = ROOT / "results" / "v2_per_family_metrics.csv"
family_df.to_csv(csv_out, index=False)
print(f"  ✓ {csv_out.name}")

# ============================================================================
# 输出报告文本片段
# ============================================================================
print()
print("=" * 70)
print("  报告 Result 部分核心数字")
print("=" * 70)
print(f"  • 测试集大小: {len(df)} (剔除 1 OOD outlier 后 n={len(df_clean)})")
print(f"  • 整体 MAE: {overall_mae:.2f} K")
print(f"  • 整体 R²:  {overall_r2:.3f}")
print(f"  • v1 baseline MAE: {V1_MAE} K")
print(f"  • v2 相对 v1: {(V1_MAE - overall_mae) / V1_MAE * 100:.1f}% 误差降低")
print()
print("  按家族 (按 MAE 升序):")
for _, row in family_df_sorted.iterrows():
    print(f"    • {row['family']:12s} (n={row['n']:3d}):  MAE = {row['mae']:.2f} K")
print()
print("=" * 70)
print(f"  全部 figures 保存到: {OUT_DIR}")
print("=" * 70)
