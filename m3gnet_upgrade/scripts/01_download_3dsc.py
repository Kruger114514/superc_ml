"""
01_download_3dsc.py — 下载 3DSC 数据集 (v2 最终方案)
===================================================

为什么用 3DSC?
  - 唯一一个 cif 完整可下载的实验超导数据集
  - 5,700 个超导体, 每个都配有从 Materials Project 匹配的 3D 结构
  - 发表在 Nature Scientific Data 10, 816 (2023), 经过同行评议
  - 数据格式标准, 易于和 M3GNet 集成

为什么不用 SuperBand?
  - csv 里 cif 列只是 ID, 完整 cif 数据不在 GitHub repo 里
  - 完整 cif 需要从 superband.work 爬取
  - 33 个示例 cif 不够训练

诚实声明 (写报告里):
  3DSC 也不包含 LaH10 等高压氢化物 (因为 Materials Project 没有它们的常压结构).
  这一限制是物理决定的, 不是项目的缺陷.
  我们的项目重点是"方法论升级 (组分 -> 结构)", 不是"预测所有超导体".

数据来源:
  - GitHub: https://github.com/aimat-lab/3DSC
  - Paper: Sommer et al., Sci Data 10, 816 (2023)

运行: python scripts/01_download_3dsc.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "3dsc"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def clone_3dsc_repo():
    """克隆 3DSC GitHub 仓库"""
    print("=" * 70)
    print("  [1/3] 克隆 3DSC GitHub 仓库")
    print("=" * 70)

    target = DATA_DIR / "3DSC_repo"
    if target.exists():
        print(f"\n  仓库已存在: {target}")
        cifs_dir = target / "superconductors_3D" / "data" / "final" / "MP" / "cifs"
        if cifs_dir.exists():
            n_cifs = len(list(cifs_dir.glob("*.cif")))
            print(f"  已有 {n_cifs} 个 cif 文件")
            if n_cifs > 100:
                print(f"  ✓ 数据完整, 跳过克隆")
                return target
        print(f"  数据不完整, 重新克隆...")
        import shutil
        shutil.rmtree(target)

    print(f"\n  克隆到: {target}")
    print(f"  (约 150-200 MB, 可能需要 5-15 分钟)")
    print()

    try:
        # --depth 1 大幅减少下载量
        result = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/aimat-lab/3DSC.git", str(target)],
            check=True, capture_output=True, text=True,
        )
        print(f"  ✓ 克隆完成")
        return target
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 克隆失败:")
        print(f"    {e.stderr}")
        print()
        print("  请手动执行:")
        print(f"    cd {DATA_DIR}")
        print(f"    git clone --depth 1 https://github.com/aimat-lab/3DSC.git 3DSC_repo")
        print()
        print("  或者用国内镜像 (国内访问 GitHub 慢时):")
        print(f"    git clone --depth 1 https://gitclone.com/github.com/aimat-lab/3DSC.git 3DSC_repo")
        return None


def explore_3dsc(repo_dir: Path):
    """探索 3DSC 仓库内容"""
    print()
    print("=" * 70)
    print("  [2/3] 探索 3DSC 仓库结构")
    print("=" * 70)

    if not repo_dir or not repo_dir.exists():
        print("  ⚠ 仓库目录不存在")
        return None

    print(f"\n  根目录: {repo_dir}")

    paths_to_check = {
        "主 csv (SC_MP.csv)":
            repo_dir / "superconductors_3D" / "data" / "final" / "MP" / "SC_MP.csv",
        "cif 文件夹":
            repo_dir / "superconductors_3D" / "data" / "final" / "MP" / "cifs",
        "README":
            repo_dir / "README.md",
    }

    for name, path in paths_to_check.items():
        if path.exists():
            if path.is_file():
                size_kb = path.stat().st_size / 1024
                print(f"  ✓ {name}: {size_kb:.1f} KB")
            else:
                n_files = len(list(path.glob("*")))
                print(f"  ✓ {name}: {n_files} 个文件")
        else:
            print(f"  ✗ {name}: 未找到")
            print(f"      路径: {path}")


def diagnose_3dsc(repo_dir: Path):
    """诊断 3DSC 关键化合物覆盖"""
    print()
    print("=" * 70)
    print("  [3/3] 数据集统计与覆盖检查")
    print("=" * 70)

    csv_path = repo_dir / "superconductors_3D" / "data" / "final" / "MP" / "SC_MP.csv"
    if not csv_path.exists():
        print(f"  ✗ 找不到主 csv")
        return

    try:
        import pandas as pd
    except ImportError:
        print("  ⚠ pandas 未装, 跳过")
        return

    df = pd.read_csv(csv_path)
    print(f"\n  总记录数: {len(df)}")
    print(f"  列名 (前 10): {list(df.columns)[:10]}")

    # 找关键列
    formula_col = None
    tc_col = None
    for c in df.columns:
        if "formula" in c.lower() and not formula_col:
            formula_col = c
        if c.lower() in ["tc", "critical_temp", "critical_temp_k"]:
            tc_col = c

    if not formula_col:
        # 退路: 试用第一列或常见名字
        for c in df.columns:
            if "comp" in c.lower() or "material" in c.lower():
                formula_col = c
                break

    print(f"  formula 列: {formula_col}")
    print(f"  Tc 列: {tc_col}")
    print()

    if not formula_col:
        print(f"  ⚠ 未找到 formula 列, 全部列: {list(df.columns)}")
        return

    formulas = df[formula_col].astype(str)

    # 家族容量
    print(f"  按家族容量 (基于化学组分关键字):")
    families = [
        ("BCS-like (Nb, V, Ta 基)",
         formulas.str.contains(r"Nb|V[A-Z]|Ta", regex=True)),
        ("Cuprate (Cu + O)",
         formulas.str.contains("Cu") & formulas.str.contains("O")),
        ("Iron-based",
         formulas.str.contains("Fe") &
         (formulas.str.contains("As") | formulas.str.contains("Se") |
          formulas.str.contains("Te") | formulas.str.contains("P"))),
        ("Nickelate (Ni + O)",
         formulas.str.contains("Ni") & formulas.str.contains("O")),
        ("含氢化合物",
         formulas.str.contains(r"H\d|H[A-Z]", regex=True)),
        ("MgB2 系 (Mg + B)",
         formulas.str.contains("Mg") & formulas.str.contains("B")),
    ]

    for name, mask in families:
        n = mask.sum()
        if n > 0 and tc_col:
            tc_sub = pd.to_numeric(df[mask][tc_col], errors="coerce").dropna()
            if len(tc_sub) > 0:
                max_tc = tc_sub.max()
                mean_tc = tc_sub.mean()
                print(f"    {name:30s}: {n:5d} 条  (max Tc = {max_tc:6.1f} K, mean = {mean_tc:5.1f} K)")
            else:
                print(f"    {name:30s}: {n:5d} 条")
        else:
            print(f"    {name:30s}: {n:5d} 条")

    # Tc 分布
    if tc_col:
        print()
        print(f"  Tc 分布:")
        tc = pd.to_numeric(df[tc_col], errors="coerce").dropna()
        print(f"    总数: {len(tc)}")
        print(f"    范围: {tc.min():.3f} – {tc.max():.1f} K")
        print(f"    中位数: {tc.median():.2f} K")
        print(f"    > 30 K: {(tc > 30).sum()}")
        print(f"    > 77 K: {(tc > 77).sum()}")
        print(f"    > 100 K: {(tc > 100).sum()}")


def main():
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "3DSC 数据集下载 (v2 最终方案)" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("3DSC 是当前最完整的实验超导数据集 (含真实可用的 3D 晶体结构).")
    print()
    print("下载内容:")
    print("  - 主索引 csv: ~1 MB")
    print("  - cif 文件: ~5,700 个 (总计 100-150 MB)")
    print()
    print("注意:")
    print("  - 第一次跑会调用 git clone, 需要联网且支持 GitHub 访问")
    print("  - 如果网络慢, 用 Ctrl+C 中断, 再用国内镜像手动 clone")
    print()

    input("按 Enter 继续 (或 Ctrl+C 退出)... ")

    repo_dir = clone_3dsc_repo()

    if repo_dir:
        explore_3dsc(repo_dir)
        diagnose_3dsc(repo_dir)

    print()
    print("=" * 70)
    print("  ✓ 3DSC 准备完成")
    print()
    print("  下一步: 跑 02_prepare_dataset_3dsc.py 准备训练数据")
    print("=" * 70)


if __name__ == "__main__":
    main()
