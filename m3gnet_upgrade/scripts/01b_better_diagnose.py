"""
01b_better_diagnose.py — 更准确的 SuperBand 数据集诊断
=====================================================

之前的诊断匹配太宽松, 导致误判. 这个脚本用更精准的方法.

运行: python scripts/01b_better_diagnose.py
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# 自动找 csv 文件 (不写死路径)
SUPERBAND_DIR = ROOT / "data" / "superband"
CSV_PATH = None
for candidate in SUPERBAND_DIR.rglob("superband.csv"):
    CSV_PATH = candidate
    break
if CSV_PATH is None:
    # 回退: 找任何 csv
    for candidate in SUPERBAND_DIR.rglob("*.csv"):
        if "license" not in candidate.name.lower():
            CSV_PATH = candidate
            print(f"使用回退 csv: {CSV_PATH}")
            break

if CSV_PATH is None or not CSV_PATH.exists():
    print(f"找不到 csv 文件 (在 {SUPERBAND_DIR})")
    print("请先跑 01_download_superband.py")
    exit(1)

print(f"读取 {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"总条数: {len(df)}")
print(f"列名: {list(df.columns)}")
print()

# 关键列分析
print("=" * 70)
print("  关键列分布")
print("=" * 70)

# Tc 分布
if "Tc" in df.columns:
    tc = pd.to_numeric(df["Tc"], errors="coerce")
    valid = tc.dropna()
    print(f"\nTc 列:")
    print(f"  有效 Tc 数: {len(valid)}")
    print(f"  Tc 范围: {valid.min():.2f} – {valid.max():.2f} K")
    print(f"  Tc 中位数: {valid.median():.2f} K")
    print(f"  Tc > 30 K (高温超导): {(valid > 30).sum()}")
    print(f"  Tc > 77 K (液氮温度): {(valid > 77).sum()}")
    print(f"  Tc > 150 K: {(valid > 150).sum()}")

# 含氢化物 (含 H 元素的实际数量)
print(f"\n含氢化合物 (formula 含 'H'):")
h_count = df["formula"].astype(str).str.contains(r"\bH\d*\b|H[A-Z]", regex=True).sum()
print(f"  数量: {h_count}")

# 含 cif 数据的条目
if "cif" in df.columns:
    cif_count = df["cif"].astype(str).str.len() > 100  # cif 文本一般都很长
    print(f"\n含 cif 结构数据的条目: {cif_count.sum()}")

# 按家族粗略分类 (基于化学组分)
print()
print("=" * 70)
print("  按家族粗略分类 (基于化学组分关键字)")
print("=" * 70)

formulas = df["formula"].astype(str)

families = {
    "Cuprate (含 Cu + O)":      formulas.str.contains("Cu") & formulas.str.contains("O"),
    "Iron-based (含 Fe + As/Se/Te/P)": formulas.str.contains("Fe") &
                                       (formulas.str.contains("As") | formulas.str.contains("Se") |
                                        formulas.str.contains("Te") | formulas.str.contains(r"\bP\d", regex=True)),
    "Nickelate (含 Ni + O)":    formulas.str.contains("Ni") & formulas.str.contains("O"),
    "Hydride (含 H, 不含 O)":   formulas.str.contains("H") & ~formulas.str.contains("O"),
    "MgB2-like (含 Mg + B)":   formulas.str.contains("Mg") & formulas.str.contains("B"),
    "Heusler/Nb 基 (含 Nb)":   formulas.str.contains("Nb"),
    "Ti 基 (含 Ti, 不含 O 不含 Cu)": formulas.str.contains("Ti") &
                                  ~formulas.str.contains("O") & ~formulas.str.contains("Cu"),
}

for family, mask in families.items():
    n = mask.sum()
    sample = df[mask]["formula"].head(5).tolist()
    if "Tc" in df.columns and n > 0:
        tc_sub = pd.to_numeric(df[mask]["Tc"], errors="coerce").dropna()
        max_tc = tc_sub.max() if len(tc_sub) > 0 else 0
        print(f"\n  {family}: {n} 条 (最高 Tc = {max_tc:.1f} K)")
        print(f"    示例: {sample[:3]}")

# 精确查找 v1 关键化合物
print()
print("=" * 70)
print("  v1 关键化合物精确查找 (考虑掺杂)")
print("=" * 70)

key_compounds = {
    "MgB2":          [r"\bMg(\d|\.|$)+B(\d|\.|$)+$"],
    "Nb3Ge":         [r"Nb3Ge$", r"Nb.+Ge.+$"],
    "Nb3Sn":         [r"Nb3Sn$", r"Nb.+Sn.+$"],
    "YBa2Cu3O":      [r"Y.+Ba2Cu3O", r"YBa2Cu3O"],
    "Bi2Sr2CaCu2O":  [r"Bi2Sr2Ca.*Cu.*O"],
    "Hg-1223":       [r"Hg.*Ba2Ca2Cu3"],
    "Tl-2223":       [r"Tl2Ba2Ca2Cu3"],
    "LaFeAsO":       [r"LaFeAsO", r"La.+Fe.+As.+O"],
    "BaFe2As2":      [r"BaFe2As2", r"Ba.+Fe2As2"],
    "FeSe":          [r"^Fe.*Se", r"FeSe"],
    "La3Ni2O7":      [r"La3Ni2O7", r"La.+Ni.+O7"],
    "NdNiO2":        [r"Nd.+Ni.*O2", r"NdNiO2"],
    "H3S":           [r"^H3S$", r"H3S\d?$"],
    "LaH10":         [r"LaH10", r"La.*H1[0-9]"],
    "YH9":           [r"YH9", r"Y.*H9"],
    "YH6":           [r"^YH6$"],
}

for name, patterns in key_compounds.items():
    found = []
    for pattern in patterns:
        matches = df[formulas.str.contains(pattern, regex=True, na=False)]
        for idx, row in matches.iterrows():
            if row["formula"] not in [f for f, _ in found]:
                tc = pd.to_numeric(pd.Series([row.get("Tc")]), errors="coerce").iloc[0]
                tc_str = f"Tc={tc:.1f}K" if pd.notna(tc) else "Tc=?"
                found.append((row["formula"], tc_str))
    if found:
        print(f"  ✓ {name}: 找到 {len(found)} 条")
        for f, t in found[:3]:
            print(f"      {f} ({t})")
        if len(found) > 3:
            print(f"      ... 共 {len(found)} 条")
    else:
        print(f"  ✗ {name}: 未找到")

print()
print("=" * 70)
print("  结论 + 下一步建议")
print("=" * 70)
print("""
1. 看 [按家族粗略分类] 部分, 了解 SuperBand 在每个家族的真实容量
2. 看 [v1 关键化合物精确查找] 确认哪些 v1 化合物在 SuperBand 里
3. 把这个输出截图发回去, 我帮你设计训练/测试划分策略

关键问题:
  - 铜氧化物的 v1 测试集化合物是否在 SuperBand 里? (应该有, 但具体哪些?)
  - 镍基 La3Ni2O7 找到了 - 这是个亮点
  - 高压氢化物 (LaH10/YH6/YH9) 大概率没有 (论文确认排除了)
""")
