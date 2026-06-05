"""
04_evaluate.py — 在测试集上评估训练好的 M3GNet 模型
=====================================================

输入: models/m3gnet_tc.pt + data/processed_3dsc/test.pkl
输出:
  - results/figures/v2_pred_vs_exp.png  (主散点图)
  - results/figures/v2_error_by_family.png (按家族箱线图)
  - results/v2_test_predictions.csv (每条测试集样本的预测)
  - results/v2_metrics.json (总体 + 按家族 R², MAE)

跟 v1 的 analyze_families.py 输出风格保持一致, 方便对比。

运行: python scripts/04_evaluate.py
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, r2_score

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# 复用 v1 的家族配色, 报告里好对照
FAMILY_COLORS = {
    "Hydride": "#dc2626",
    "Cuprate": "#10b981",
    "Iron-based": "#f59e0b",
    "Nickelate": "#a855f7",
    "Other": "#3b82f6",  # 包括 BCS + 其他
}


def load_model(config_path: Path, ckpt_path: Path):
    """加载训练好的 Lightning checkpoint, 返回可推理的模型"""
    # 局部 import 避免脚本顶层耦合
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.models import M3GNet

    # 这里假设和 03_train_m3gnet_tc.py 的 TcRegressor 兼容
    # 简化处理: 直接重建模型, 加载 state_dict
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    # 动态加载 03 脚本以复用 TcRegressor 类
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_module",
        ROOT / "scripts" / "03_train_m3gnet_tc.py",
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = train_mod.build_model(cfg)
    lit_module = train_mod.TcRegressor.load_from_checkpoint(
        ckpt_path, model=model,
    )
    lit_module.eval()
    return lit_module, cfg


def predict_all(lit_module, records, cfg):
    """逐条预测 (因为评估不需要 batch)"""
    from functools import partial
    from matgl.config import DEFAULT_ELEMENTS
    from matgl.ext.pymatgen import Structure2Graph
    from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph

    converter = Structure2Graph(
        element_types=DEFAULT_ELEMENTS,
        cutoff=cfg["model"]["cutoff"],
    )
    structures = [r["structure"] for r in records]
    tc_values = [r["tc"] for r in records]
    log_tf = cfg["data"]["log_transform"]
    if log_tf:
        labels = {"tc": [float(np.log(t + 1)) for t in tc_values]}
    else:
        labels = {"tc": [float(t) for t in tc_values]}

    ds = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels=labels,
        include_line_graph=True,
    )
    collate = partial(collate_fn_graph, include_line_graph=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=16, collate_fn=collate, shuffle=False,
    )

    preds = []
    device = next(lit_module.parameters()).device
    with torch.no_grad():
        for batch in loader:
            g, lattice, lg, state_attr, _ = batch
            g = g.to(device)
            lg = lg.to(device) if lg is not None else None
            p = lit_module(g, lg, state_attr, lattice).squeeze(-1).cpu().numpy()
            preds.extend(p.tolist())

    preds = np.array(preds)
    if log_tf:
        preds = np.exp(preds) - 1
    return preds


def evaluate_overall(true_tc, pred_tc):
    r2 = r2_score(true_tc, pred_tc)
    mae = mean_absolute_error(true_tc, pred_tc)
    return {"r2": float(r2), "mae": float(mae), "n": len(true_tc)}


def evaluate_by_family(df):
    """按家族分组计算 R², MAE"""
    stats = []
    for family, grp in df.groupby("family"):
        if len(grp) < 2:  # R² 需要至少 2 个点
            stats.append({
                "family": family, "n": len(grp),
                "mae": float(mean_absolute_error(grp["true_tc"], grp["pred_tc"])),
                "r2": None,
            })
        else:
            stats.append({
                "family": family, "n": len(grp),
                "mae": float(mean_absolute_error(grp["true_tc"], grp["pred_tc"])),
                "r2": float(r2_score(grp["true_tc"], grp["pred_tc"])),
            })
    return stats


def plot_pred_vs_exp(df, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 7))
    for family, grp in df.groupby("family"):
        ax.scatter(grp["true_tc"], grp["pred_tc"],
                   color=FAMILY_COLORS.get(family, "gray"),
                   label=f"{family} (n={len(grp)})",
                   alpha=0.7, s=60, edgecolor="black", linewidth=0.5)
    max_tc = max(df["true_tc"].max(), df["pred_tc"].max()) * 1.05
    ax.plot([0, max_tc], [0, max_tc], "--", color="gray", label="Perfect prediction")
    ax.fill_between([0, max_tc], [0, max_tc * 0.7], [0, max_tc * 1.3],
                    color="gray", alpha=0.1, label="±30% band")
    ax.set_xlim(-5, max_tc)
    ax.set_ylim(-5, max_tc)
    ax.set_xlabel(r"Experimental $T_c$ (K)", fontsize=12)
    ax.set_ylabel(r"Predicted $T_c$ (K)", fontsize=12)
    ax.set_title("v2 (M3GNet) — Predicted vs Experimental Tc", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path.name}")


def plot_error_by_family(df, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    families_order = sorted(df["family"].unique(),
                            key=lambda f: df[df["family"] == f]["error"].abs().mean())
    data = [df[df["family"] == f]["error"].values for f in families_order]
    colors = [FAMILY_COLORS.get(f, "gray") for f in families_order]
    bp = ax.boxplot(data, labels=families_order, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    # 把每个点也散点画上去
    for i, (f, color) in enumerate(zip(families_order, colors), 1):
        sub = df[df["family"] == f]
        x = np.random.normal(i, 0.04, size=len(sub))
        ax.scatter(x, sub["error"], color=color, edgecolor="black",
                   linewidth=0.5, s=30, alpha=0.6, zorder=3)
    ax.axhline(0, color="red", linestyle="--", alpha=0.7, label="Zero error")
    ax.set_ylabel("Prediction error: Predicted − Experimental (K)", fontsize=11)
    ax.set_title("v2 (M3GNet) — Error distribution by family", fontsize=13)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path.name}")


def main():
    print("=" * 70)
    print("  Evaluate M3GNet Tc Model")
    print("=" * 70)

    config_path = ROOT / "configs" / "train_config.yaml"
    ckpt_path = ROOT / "models" / "m3gnet_tc.pt"
    test_path = ROOT / "data" / "processed_3dsc" / "test.pkl"

    for p in [config_path, ckpt_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}, 请先跑相应的准备/训练脚本")

    print(f"\n[1/4] 加载模型 {ckpt_path}")
    lit_module, cfg = load_model(config_path, ckpt_path)

    print(f"[2/4] 加载测试集 {test_path}")
    with open(test_path, "rb") as f:
        test_records = pickle.load(f)
    print(f"      共 {len(test_records)} 条")

    print("[3/4] 预测...")
    pred_tc = predict_all(lit_module, test_records, cfg)
    true_tc = np.array([r["tc"] for r in test_records])
    families = [r["family"] for r in test_records]
    formulas = [r["formula"] for r in test_records]

    df = pd.DataFrame({
        "formula": formulas,
        "family": families,
        "true_tc": true_tc,
        "pred_tc": pred_tc,
        "error": pred_tc - true_tc,
        "abs_error": np.abs(pred_tc - true_tc),
    })

    print("[4/4] 计算指标 + 画图...")
    overall = evaluate_overall(true_tc, pred_tc)
    family_stats = evaluate_by_family(df)

    print()
    print(f"  整体表现: R² = {overall['r2']:.3f}, MAE = {overall['mae']:.2f} K")
    print()
    print(f"  {'Family':<15} {'n':>5} {'MAE (K)':>10} {'R²':>10}")
    print(f"  {'-'*42}")
    for s in sorted(family_stats, key=lambda x: x["mae"]):
        r2_str = f"{s['r2']:.3f}" if s["r2"] is not None else "  N/A"
        print(f"  {s['family']:<15} {s['n']:>5} {s['mae']:>10.2f} {r2_str:>10}")

    # 保存
    df.to_csv(ROOT / "results" / "v2_test_predictions.csv", index=False)
    with open(ROOT / "results" / "v2_metrics.json", "w") as f:
        json.dump({"overall": overall, "by_family": family_stats}, f, indent=2)

    plot_pred_vs_exp(df, FIG_DIR / "v2_pred_vs_exp.png")
    plot_error_by_family(df, FIG_DIR / "v2_error_by_family.png")

    print()
    print("=" * 70)
    print("  评估完成. 下一步:")
    print("  - python scripts/05_predict_polymorphs.py  (多形态对比)")
    print("  - python scripts/06_compare_with_v1.py     (v1 vs v2 对比表)")
    print("=" * 70)


if __name__ == "__main__":
    main()
