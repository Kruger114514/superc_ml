"""
01d_inspect_cif_column.py — 看一眼 csv 里 cif 列到底是什么
==========================================================

cif 列长度都 <= 500, 不像真的 cif 结构. 那它到底是什么?
- cif 文件路径?
- cif 文件名?
- Materials Project ID?
- 空字符串?

运行: python scripts/01d_inspect_cif_column.py
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SUPERBAND_DIR = ROOT / "data" / "superband"

# 找 csv
CSV_PATH = None
for c in SUPERBAND_DIR.rglob("superband.csv"):
    CSV_PATH = c
    break
if CSV_PATH is None:
    print("找不到 superband.csv")
    exit(1)

print(f"读取 {CSV_PATH}\n")
df = pd.read_csv(CSV_PATH)


# ============================================================
# Section 1: cif 列的实际样子
# ============================================================
print("=" * 70)
print("  cif 列样本检查")
print("=" * 70)

cif_strs = df["cif"].astype(str)
cif_lens = cif_strs.str.len()

print(f"\ncif 长度分布:")
print(f"  最短: {cif_lens.min()}")
print(f"  最长: {cif_lens.max()}")
print(f"  中位数: {cif_lens.median():.1f}")

print(f"\ncif 长度直方图:")
bins = [0, 5, 10, 20, 50, 100, 200, 500, 99999]
for lo, hi in zip(bins[:-1], bins[1:]):
    n = ((cif_lens >= lo) & (cif_lens < hi)).sum()
    print(f"  长度 [{lo:5d}, {hi:5d}): {n} 条")

# 看 10 个不同长度的 cif 实际内容
print(f"\n10 个示例 (formula | cif_len | cif 内容前 100 字符):")
print("-" * 70)
sample_indices = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 7000, 8000]
for i in sample_indices:
    if i < len(df):
        row = df.iloc[i]
        f = str(row.get('formula', '?'))[:25]
        c = str(row.get('cif', ''))[:100]
        cl = len(str(row.get('cif', '')))
        print(f"  [{i:4d}] {f:25s} | len={cl:5d} | {c!r}")


# ============================================================
# Section 2: cif 文件夹独立检查
# ============================================================
print()
print("=" * 70)
print("  独立的 cif/ 文件夹")
print("=" * 70)

# repo 里有个独立的 cif/ 文件夹, 33 个文件
for repo_dir in SUPERBAND_DIR.glob("*"):
    if not repo_dir.is_dir():
        continue
    cif_dir = repo_dir / "cif"
    if cif_dir.exists():
        cif_files = list(cif_dir.glob("*.cif"))
        print(f"  找到 cif 文件夹: {cif_dir}")
        print(f"  cif 文件数: {len(cif_files)}")
        if cif_files:
            # 看一个 cif 真实结构
            sample_cif = cif_files[0]
            print(f"\n  示例 cif 文件: {sample_cif.name}")
            print(f"  大小: {sample_cif.stat().st_size} 字节")
            print(f"  前 20 行:")
            with open(sample_cif, encoding='utf-8', errors='ignore') as fh:
                for i, line in enumerate(fh):
                    if i >= 20:
                        break
                    print(f"    {line.rstrip()}")

            # 列出几个 cif 文件名
            print(f"\n  前 10 个 cif 文件名:")
            for c in cif_files[:10]:
                print(f"    {c.name}")


# ============================================================
# Section 3: 其他文件
# ============================================================
print()
print("=" * 70)
print("  整个 SuperBand_repo 结构")
print("=" * 70)

for repo_dir in SUPERBAND_DIR.glob("SuperBand*"):
    if not repo_dir.is_dir():
        continue
    print(f"\n  根目录: {repo_dir}")
    for item in sorted(repo_dir.iterdir()):
        if item.is_file():
            sz = item.stat().st_size
            sz_str = f"{sz/1024/1024:.1f}MB" if sz > 1024*1024 else f"{sz/1024:.1f}KB"
            print(f"    📄 {item.name} ({sz_str})")
        elif item.is_dir():
            n = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"    📁 {item.name}/ ({n} 文件)")


print()
print("=" * 70)
print("  这能告诉我们什么")
print("=" * 70)
print("""
看完上面的输出, 我可以判断:

如果 cif 列是 "10001" 这种数字 -> 它是文件ID, 真实 cif 在 cif/ 文件夹
如果 cif 列是 "structures/xxx.cif" -> 是相对路径
如果 cif 列是空或 "nan" -> 没有结构数据
如果 cif/ 文件夹里只有 33 个文件, 那大部分超导体不带结构

把这个输出截图发我, 我立刻给出下一步的最终策略!
""")
