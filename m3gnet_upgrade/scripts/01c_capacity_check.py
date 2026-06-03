"""
01c_capacity_check.py — SuperBand 容量与质量评估
================================================

回答几个关键问题:
1. 8625 条里, 有多少条有 'cif' 列里的真实结构数据?
2. 各家族的真实容量是多少 (粗略分类)
3. LaH10、La3Ni2O7 等关键化合物的 cif 长什么样? 是否高压相?
4. Tc 分布

运行: python scripts/01c_capacity_check.py
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# 自动找 csv
SUPERBAND_DIR = ROOT / "data" / "superband"
CSV_PATH = None
for c in SUPERBAND_DIR.rglob("superband.csv"):
    CSV_PATH = c
    break
if CSV_PATH is None:
    print("找不到 superband.csv")
    exit(1)

print(f"读取 {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"总条数: {len(df)}\n")


# ============================================================
# Section 1: cif 列质量检查
# ============================================================
print("=" * 70)
print("  [1/4] cif 结构数据质量")
print("=" * 70)

if "cif" not in df.columns:
    print("  ⚠ 没有 'cif' 列")
else:
    cif_strs = df["cif"].astype(str)
    cif_lens = cif_strs.str.len()

    print(f"  cif 列条目数: {len(cif_strs)}")
    print(f"  cif 长度统计:")
    print(f"    最小: {cif_lens.min()}")
    print(f"    最大: {cif_lens.max()}")
    print(f"    中位数: {cif_lens.median():.0f}")
    print(f"    平均: {cif_lens.mean():.0f}")

    # cif 至少 500 字符才算"有真实数据"
    has_real_cif = (cif_lens > 500).sum()
    nan_cif = (cif_strs == "nan").sum()
    print(f"\n  含真实 cif (>500 字符) 的条目: {has_real_cif} ({has_real_cif/len(df)*100:.1f}%)")
    print(f"  cif 为 nan 的条目: {nan_cif}")

    # 看一个示例
    sample = df[cif_lens > 500].iloc[0]
    print(f"\n  cif 示例 (formula={sample['formula']}, Tc={sample.get('Tc', '?')}):")
    cif_lines = str(sample['cif']).split('\n')[:8]
    for line in cif_lines:
        print(f"    {line[:80]}")
    print(f"    ... (共 {len(str(sample['cif']).split(chr(10)))} 行)")


# ============================================================
# Section 2: Tc 分布
# ============================================================
print()
print("=" * 70)
print("  [2/4] Tc 分布")
print("=" * 70)

if "Tc" in df.columns:
    tc = pd.to_numeric(df["Tc"], errors="coerce").dropna()
    print(f"  有效 Tc: {len(tc)} 条")
    print(f"  Tc 统计:")
    print(f"    min:    {tc.min():.3f} K")
    print(f"    max:    {tc.max():.1f} K  ⭐")
    print(f"    median: {tc.median():.2f} K")
    print(f"    mean:   {tc.mean():.2f} K")

    bins = [
        ("Tc <= 1", (tc <= 1).sum()),
        ("1 < Tc <= 10", ((tc > 1) & (tc <= 10)).sum()),
        ("10 < Tc <= 30", ((tc > 10) & (tc <= 30)).sum()),
        ("30 < Tc <= 77 (高温)", ((tc > 30) & (tc <= 77)).sum()),
        ("77 < Tc <= 150", ((tc > 77) & (tc <= 150)).sum()),
        ("150 < Tc <= 200", ((tc > 150) & (tc <= 200)).sum()),
        ("Tc > 200 (高压?)", (tc > 200).sum()),
    ]
    print(f"\n  Tc 分布:")
    total = len(tc)
    for name, count in bins:
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:30s} {count:5d} ({pct:5.1f}%) {bar}")


# ============================================================
# Section 3: 高 Tc 化合物详情 (Tc > 100K 都列出来)
# ============================================================
print()
print("=" * 70)
print("  [3/4] 高 Tc 化合物详情 (Tc > 150 K)")
print("=" * 70)

if "Tc" in df.columns and "formula" in df.columns:
    tc_num = pd.to_numeric(df["Tc"], errors="coerce")
    high_tc = df[tc_num > 150].copy()
    high_tc["Tc_num"] = tc_num[tc_num > 150]
    high_tc = high_tc.sort_values("Tc_num", ascending=False)
    print(f"  Tc > 150 K 的条目: {len(high_tc)}")
    print()
    if len(high_tc) > 0:
        for i, row in high_tc.head(30).iterrows():
            cif_len = len(str(row.get("cif", "")))
            has_cif = "✓" if cif_len > 500 else "✗"
            sg = row.get("SG_syb", "?")
            print(f"    {has_cif} {row['formula']:30s}  Tc={row['Tc_num']:6.1f}K  "
                  f"SG={str(sg)[:15]:15s}  cif_len={cif_len}")


# ============================================================
# Section 4: 家族容量估算 (基于化学组分关键字)
# ============================================================
print()
print("=" * 70)
print("  [4/4] 家族容量估算")
print("=" * 70)

formulas = df["formula"].astype(str)
tc_num = pd.to_numeric(df["Tc"], errors="coerce")
cif_lens = df["cif"].astype(str).str.len() if "cif" in df.columns else pd.Series([0] * len(df))

family_rules = [
    ("Cuprate (Cu+O)",
     formulas.str.contains("Cu") & formulas.str.contains("O")),
    ("Iron-based (Fe + As/Se/Te/P)",
     formulas.str.contains("Fe") &
     (formulas.str.contains("As") | formulas.str.contains("Se") |
      formulas.str.contains("Te") | formulas.str.contains(r"P\d", regex=True))),
    ("Nickelate (Ni + O)",
     formulas.str.contains("Ni") & formulas.str.contains("O")),
    ("Hydride (含 H)",
     formulas.str.contains(r"H\d|H[A-Z]", regex=True)),
    ("MgB2 系 (Mg + B)",
     formulas.str.contains("Mg") & formulas.str.contains("B")),
    ("A15 (Nb3X or V3X)",
     formulas.str.contains(r"Nb3|V3", regex=True)),
    ("Heavy fermion (含 U)",
     formulas.str.contains(r"U[A-Z]|^U\d", regex=True)),
    ("Organic (含 C 不含 O)",
     formulas.str.contains("C") & ~formulas.str.contains("O") &
     ~formulas.str.contains("Cu") & ~formulas.str.contains("Ca") &
     ~formulas.str.contains("Cd") & ~formulas.str.contains("Cs") &
     ~formulas.str.contains("Cr")),
]

print(f"  {'家族':40s} {'总数':>6s} {'含 cif':>8s} {'max Tc':>10s}")
print("  " + "-" * 68)
for family, mask in family_rules:
    n = mask.sum()
    n_cif = ((mask) & (cif_lens > 500)).sum()
    max_tc = tc_num[mask].max() if mask.sum() > 0 else None
    max_tc_str = f"{max_tc:.1f}K" if pd.notna(max_tc) else "?"
    print(f"  {family:40s} {n:>6d} {n_cif:>8d} {max_tc_str:>10s}")


# ============================================================
# 结论
# ============================================================
print()
print("=" * 70)
print("  结论 + 后续策略")
print("=" * 70)
print("""
看完上面四个 section, 我能根据结果给你定下:
1. 训练集大小 (有 cif 的条目数)
2. 各家族应该用多少做训练 / 测试
3. 高 Tc 离群点的处理策略 (LaH10/YH9 要不要训练?)

请把这个输出的 [1/4] [3/4] [4/4] 三段截图发回, 我马上给出训练配置.
""")
