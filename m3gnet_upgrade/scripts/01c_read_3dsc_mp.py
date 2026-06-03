"""
01c_read_3dsc_mp.py — 正确读取 3DSC_MP.csv 主表
==================================================

3DSC_MP.csv 的第一行是注释 (以 # 开头), 直接用 pandas 读会出错.
这个脚本正确处理这种情况, 显示真实的列名和数据.

运行: python scripts/01c_read_3dsc_mp.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = (
    ROOT / "data" / "3dsc" / "3DSC_repo"
    / "superconductors_3D" / "data" / "final" / "MP" / "3DSC_MP.csv"
)

if not CSV_PATH.exists():
    print(f"找不到 {CSV_PATH}")
    exit(1)

print(f"读取: {CSV_PATH}")
print(f"文件大小: {CSV_PATH.stat().st_size / 1024 / 1024:.1f} MB")
print()

# 关键: skiprows=1 跳过第一行注释
df = pd.read_csv(CSV_PATH, skiprows=1)

print("=" * 70)
print("  基本信息")
print("=" * 70)
print(f"  总行数: {len(df)}")
print(f"  总列数: {len(df.columns)}")
print()
print(f"  全部列名:")
for i, c in enumerate(df.columns):
    dtype = df[c].dtype
    n_valid = df[c].notna().sum()
    print(f"    [{i:3d}] {c:50s} (dtype={dtype}, n_valid={n_valid})")


print()
print("=" * 70)
print("  寻找关键列")
print("=" * 70)

# 找化学式列 + Tc 列 + cif 路径列
formula_col = None
tc_col = None
cif_col = None
mpid_col = None

for c in df.columns:
    cl = c.lower()
    if "formula" in cl and not formula_col:
        formula_col = c
    elif "name" == cl and not formula_col:
        formula_col = c
    if cl in ["tc", "critical_temp", "critical_temp_k", "t_c"]:
        tc_col = c
    if "cif" in cl and "path" in cl:
        cif_col = c
    elif cl == "cif" and not cif_col:
        cif_col = c
    if "mp" in cl and ("id" in cl or "_id" == cl[-3:]):
        mpid_col = c

print(f"  化学式列: {formula_col}")
print(f"  Tc 列:    {tc_col}")
print(f"  cif 列:   {cif_col}")
print(f"  MP-id 列: {mpid_col}")


print()
print("=" * 70)
print("  数据样本 (前 5 行)")
print("=" * 70)
# 只显示关键列
key_cols = [c for c in [formula_col, tc_col, cif_col, mpid_col] if c]
if key_cols:
    print(df[key_cols].head().to_string())
else:
    print(df.head().to_string())


print()
print("=" * 70)
print("  Tc 分布")
print("=" * 70)
if tc_col:
    tc = pd.to_numeric(df[tc_col], errors="coerce").dropna()
    print(f"  有效 Tc 数: {len(tc)}")
    print(f"  范围: {tc.min():.3f} - {tc.max():.1f} K")
    print(f"  中位数: {tc.median():.2f} K")
    print(f"  平均: {tc.mean():.2f} K")
    print()
    for lo, hi, label in [
        (0, 1, "Tc <= 1 K"),
        (1, 10, "1 < Tc <= 10 K"),
        (10, 30, "10 < Tc <= 30 K (BCS 常规)"),
        (30, 77, "30 < Tc <= 77 K (高温前)"),
        (77, 150, "77 < Tc <= 150 K (高温)"),
        (150, 9999, "Tc > 150 K (高压氢化物?)"),
    ]:
        n = ((tc > lo) & (tc <= hi)).sum()
        bar = "█" * int(n / 30)
        print(f"  {label:35s} {n:5d}  {bar}")


print()
print("=" * 70)
print("  按家族粗略分类")
print("=" * 70)

if formula_col:
    formulas = df[formula_col].astype(str)
    families = [
        ("Cuprate (Cu+O)",
         formulas.str.contains("Cu") & formulas.str.contains("O")),
        ("Iron-based (Fe + As/Se/Te/P)",
         formulas.str.contains("Fe") &
         (formulas.str.contains("As") | formulas.str.contains("Se") |
          formulas.str.contains("Te") | formulas.str.contains(r"P\d", regex=True))),
        ("Nickelate (Ni+O)",
         formulas.str.contains("Ni") & formulas.str.contains("O")),
        ("Hydride (含 H)",
         formulas.str.contains(r"H\d|H[A-Z]|^H", regex=True)),
        ("MgB2 系",
         formulas.str.contains("Mg") & formulas.str.contains("B")),
        ("A15 (Nb3X/V3X)",
         formulas.str.contains(r"Nb3|V3", regex=True)),
    ]
    print(f"\n  {'家族':40s} {'数量':>6s} {'max Tc':>10s}")
    print("  " + "-" * 60)
    for name, mask in families:
        n = mask.sum()
        if n > 0 and tc_col:
            tc_sub = pd.to_numeric(df[mask][tc_col], errors="coerce").dropna()
            if len(tc_sub) > 0:
                max_tc = tc_sub.max()
                print(f"  {name:40s} {n:>6d} {max_tc:>10.1f} K")
            else:
                print(f"  {name:40s} {n:>6d} {'?':>10s}")
        else:
            print(f"  {name:40s} {n:>6d}")


print()
print("=" * 70)
print("  cif 文件链接检查")
print("=" * 70)

# 找几个 cif 路径, 看看格式
cifs_dir = ROOT / "data" / "3dsc" / "3DSC_repo" / "superconductors_3D" / "data" / "final" / "MP" / "cifs"
if cifs_dir.exists():
    cif_files = list(cifs_dir.glob("*.cif"))
    print(f"  cif/ 文件夹: {cifs_dir}")
    print(f"  共 {len(cif_files)} 个 cif 文件")
    if cif_files:
        print(f"\n  前 5 个 cif 文件名:")
        for f in cif_files[:5]:
            print(f"    {f.name}")
else:
    print(f"  ⚠ cif 文件夹不存在")
    # 找别的可能位置
    print(f"  搜索 cif 文件...")
    repo = ROOT / "data" / "3dsc" / "3DSC_repo"
    all_cif_dirs = set()
    for c in repo.rglob("*.cif"):
        all_cif_dirs.add(c.parent)
    print(f"  找到 cif 在这些文件夹:")
    for d in sorted(all_cif_dirs):
        n = len(list(d.glob("*.cif")))
        print(f"    {d.relative_to(repo)}: {n} 个 cif")

print()
print("=" * 70)
print("  ✓ 诊断完成")
print("=" * 70)
print("""
关键信息已经全部找到, 我可以开始写下一步的训练脚本.
请把这个输出截图发回去 (特别是 [基本信息] [Tc 分布] [按家族粗略分类] 三段).
""")
