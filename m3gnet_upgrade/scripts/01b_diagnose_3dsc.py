"""
01b_diagnose_3dsc.py — 3DSC 真实诊断 (自动找 csv)
==================================================

01_download_3dsc.py 把仓库克隆下来了, 但 csv 路径找不到.
这个脚本会自动搜索整个仓库, 找到所有 csv, 分析每个 csv 的内容.

运行: python scripts/01b_diagnose_3dsc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = ROOT / "data" / "3dsc" / "3DSC_repo"

if not REPO_DIR.exists():
    print(f"找不到 {REPO_DIR}")
    print("请先跑 01_download_3dsc.py")
    sys.exit(1)


print("=" * 70)
print("  [1/3] 仓库整体结构")
print("=" * 70)

# 列出根目录的所有顶层文件夹
print(f"\n  根目录: {REPO_DIR}\n")
for item in sorted(REPO_DIR.iterdir()):
    if item.name.startswith("."):
        continue
    if item.is_dir():
        n_files = sum(1 for _ in item.rglob("*") if _.is_file())
        n_cifs = sum(1 for _ in item.rglob("*.cif"))
        n_csvs = sum(1 for _ in item.rglob("*.csv"))
        print(f"  📁 {item.name}/  ({n_files} 文件, {n_cifs} cif, {n_csvs} csv)")
    else:
        size_kb = item.stat().st_size / 1024
        print(f"  📄 {item.name}  ({size_kb:.1f} KB)")


print()
print("=" * 70)
print("  [2/3] 所有 csv 文件清单")
print("=" * 70)
print()

csvs = sorted(REPO_DIR.rglob("*.csv"))
print(f"  找到 {len(csvs)} 个 csv 文件:\n")

for i, csv in enumerate(csvs):
    rel = csv.relative_to(REPO_DIR)
    size_kb = csv.stat().st_size / 1024
    print(f"  [{i:2d}] {rel}  ({size_kb:.1f} KB)")


print()
print("=" * 70)
print("  [3/3] 分析每个 csv 的内容")
print("=" * 70)

try:
    import pandas as pd
except ImportError:
    print("  ⚠ pandas 未装, 跳过分析")
    sys.exit(0)

for csv in csvs:
    rel = csv.relative_to(REPO_DIR)
    print(f"\n  --- {rel} ---")
    try:
        # 先尝试读取一小部分看看格式
        df_preview = pd.read_csv(csv, nrows=5)
        print(f"  列数: {len(df_preview.columns)}")
        print(f"  列名 (前 15): {list(df_preview.columns)[:15]}")

        # 读全表统计
        df = pd.read_csv(csv)
        print(f"  总行数: {len(df)}")

        # 找化学式 + Tc 列
        formula_col = None
        tc_col = None
        for c in df.columns:
            cl = c.lower()
            if "formula" in cl or "composition" in cl or "material" in cl:
                if not formula_col:
                    formula_col = c
            if cl in ["tc", "critical_temp", "critical_temp_k", "t_c"]:
                tc_col = c

        if formula_col:
            print(f"  formula 列: '{formula_col}'  (示例: {df[formula_col].head(3).tolist()})")
        if tc_col:
            tc_vals = pd.to_numeric(df[tc_col], errors="coerce").dropna()
            print(f"  Tc 列: '{tc_col}'  (range: {tc_vals.min():.2f} - {tc_vals.max():.2f} K, n_valid: {len(tc_vals)})")

        # 如果这个 csv 看起来是主表 (有 formula + tc), 标记
        if formula_col and tc_col:
            print(f"  ⭐ 这是个有效的主表 (含 formula + Tc)")

    except Exception as e:
        print(f"  ✗ 读取失败: {e}")


print()
print("=" * 70)
print("  下一步")
print("=" * 70)
print("""
1. 从上面的列表里找出标 "⭐ 这是个有效的主表" 的 csv
2. 把这个 csv 的完整路径告诉我 (或者它的相对路径)
3. 我会改 02_prepare_dataset_3dsc.py 用正确路径
""")
