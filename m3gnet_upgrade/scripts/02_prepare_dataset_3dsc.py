"""
02_prepare_dataset_3dsc.py — 准备 3DSC 训练数据
================================================

把 3DSC_MP.csv 里的 5773 条记录转换为 M3GNet 训练格式:
1. 读取主表 (跳过第一行注释)
2. 每条记录加载对应的 cif 文件 -> pymatgen.Structure
3. 按家族 + 晶系做分层划分 (train 80% / val 10% / test 10%)
4. 保存为 pickle, 训练时快速加载

输入: data/3dsc/3DSC_repo/superconductors_3D/data/final/MP/
       ├── 3DSC_MP.csv  (主表, 5773 条)
       └── cifs/        (10904 个 cif 文件)

输出: data/processed_3dsc/
       ├── train.pkl   (~4600 条)
       ├── val.pkl     (~580 条)
       ├── test.pkl    (~580 条)
       ├── stats.csv   (统计信息)
       └── families.csv (家族分布)

运行: python scripts/02_prepare_dataset_3dsc.py
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "3dsc" / "3DSC_repo" / "superconductors_3D" / "data" / "final" / "MP"
CSV_PATH = DATA_DIR / "3DSC_MP.csv"
CIFS_DIR = DATA_DIR / "cifs"

OUT_DIR = ROOT / "data" / "processed_3dsc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 家族标注 (基于化学组分启发式)
# =============================================================================

def family_of(formula: str) -> str:
    """根据化学式启发式判断超导体家族"""
    f = formula.lower()

    # 优先级: 氢化物先判 (避免被 Cuprate 抢)
    if "h" in f and not "halogen" in f:
        # 但只有真的含 H 元素 (不是 Hg, Hf, He 等)
        # 检查 H 后面是否跟数字或大写字母
        import re
        if re.search(r"h\d|h[A-Z]|^h\b", formula):
            return "Hydride"

    # 镍基 (Ni 氧化物)
    if "ni" in f and "o" in f:
        return "Nickelate"

    # 铜氧化物
    if "cu" in f and "o" in f:
        return "Cuprate"

    # 铁基
    if "fe" in f and any(e in f for e in ["as", "se", "te"]):
        return "Iron-based"
    if "fe" in f and "p" in f and not "pb" in f:
        return "Iron-based"

    # MgB2 系
    if "mg" in f and "b" in f and not any(e in f for e in ["br", "ba", "bi"]):
        return "MgB2-like"

    # A15 结构 (Nb3X, V3X)
    if "nb3" in f or "v3" in f:
        return "A15"

    # 默认: BCS 类金属
    return "BCS-Other"


# =============================================================================
# 加载 cif 文件
# =============================================================================

def load_structure_safe(cif_relative_path: str, repo_dir: Path) -> Structure | None:
    """
    安全加载 cif. 失败返回 None.
    
    cif_relative_path 形如 "data/final/MP/cifs/xxx.cif" (相对于 superconductors_3D/)
    我们要拼接出绝对路径.
    """
    # 拼绝对路径: repo_dir/superconductors_3D/<cif_relative_path>
    abs_path = repo_dir / "superconductors_3D" / cif_relative_path

    if not abs_path.exists():
        # 备用: 用文件名直接在 cifs/ 文件夹里找
        filename = Path(cif_relative_path).name
        abs_path = CIFS_DIR / filename

    if not abs_path.exists():
        return None

    try:
        structure = Structure.from_file(str(abs_path))
        return structure
    except Exception as e:
        return None


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("=" * 70)
    print("  Prepare 3DSC Dataset for M3GNet Training")
    print("=" * 70)

    # ---- Step 1: 读 csv ----
    print(f"\n[1/5] 读取主表 {CSV_PATH.name}")
    if not CSV_PATH.exists():
        print(f"  ✗ 找不到 {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, skiprows=1)
    print(f"  ✓ 加载 {len(df)} 条记录")

    # 只保留需要的列
    keep_cols = ["formula_sc", "tc", "cif", "spacegroup_2", "crystal_system_2",
                 "material_id_2", "nsites_2", "cell_volume_2",
                 "cubic", "hexagonal", "tetragonal", "orthorhombic"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].rename(columns={
        "formula_sc": "formula",
        "spacegroup_2": "spacegroup",
        "crystal_system_2": "crystal_system",
        "material_id_2": "mp_id",
        "nsites_2": "n_sites",
        "cell_volume_2": "volume",
        "cif": "cif_path",
    })

    # ---- Step 2: 加家族标注 ----
    print(f"\n[2/5] 标注超导家族")
    df["family"] = df["formula"].apply(family_of)
    family_counts = df["family"].value_counts()
    print(f"  家族分布:")
    for fam, n in family_counts.items():
        tc_sub = df[df["family"] == fam]["tc"]
        max_tc = tc_sub.max()
        print(f"    {fam:20s}: {n:5d}  (max Tc = {max_tc:.1f} K)")

    # ---- Step 3: 加载 cif 文件 ----
    print(f"\n[3/5] 加载 cif 文件 (这一步比较慢, 估计 5-10 分钟)")
    repo_dir = ROOT / "data" / "3dsc" / "3DSC_repo"

    records = []
    failed = 0
    for i, row in df.iterrows():
        if (i + 1) % 500 == 0:
            print(f"  进度: {i+1}/{len(df)} (成功 {len(records)}, 失败 {failed})")

        structure = load_structure_safe(row["cif_path"], repo_dir)
        if structure is None:
            failed += 1
            continue

        records.append({
            "formula": row["formula"],
            "tc": float(row["tc"]),
            "family": row["family"],
            "spacegroup": row.get("spacegroup", "unknown"),
            "crystal_system": row.get("crystal_system", "unknown"),
            "mp_id": row.get("mp_id", ""),
            "structure": structure,
            "n_atoms": len(structure),
            "volume": float(structure.volume),
        })

    print(f"\n  ✓ 成功加载 {len(records)} / {len(df)} 条 (失败 {failed})")
    if len(records) == 0:
        print("  ✗ 没有成功加载任何 cif, 退出")
        sys.exit(1)

    # ---- Step 4: 分层划分 ----
    print(f"\n[4/5] 分层划分 train/val/test (80/10/10)")

    # 按家族分层
    families = [r["family"] for r in records]

    # 先分出 test (10%)
    train_val, test = train_test_split(
        records, test_size=0.10, stratify=families, random_state=42,
    )
    # 从 train_val 再分出 val (~11.1% of train_val = 10% of total)
    tv_families = [r["family"] for r in train_val]
    train, val = train_test_split(
        train_val, test_size=0.111, stratify=tv_families, random_state=42,
    )

    print(f"\n  集合大小:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        sizes = pd.Series([r["family"] for r in split]).value_counts().to_dict()
        print(f"    {name:5s}: {len(split):5d} 条")
        for fam, n in sorted(sizes.items(), key=lambda x: -x[1]):
            print(f"      {fam:20s}: {n}")

    # ---- Step 5: 保存 ----
    print(f"\n[5/5] 保存到 {OUT_DIR}")

    for name, data in [("train", train), ("val", val), ("test", test)]:
        out = OUT_DIR / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(data, f)
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"  ✓ {name}.pkl  ({len(data)} 条, {size_mb:.1f} MB)")

    # 保存统计 csv (不含 structure 对象)
    stats_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "structure"}
        for r in records
    ])
    stats_df.to_csv(OUT_DIR / "stats.csv", index=False)
    print(f"  ✓ stats.csv")

    family_counts.to_csv(OUT_DIR / "families.csv", header=["count"])
    print(f"  ✓ families.csv")

    print()
    print("=" * 70)
    print(f"  ✓ 数据准备完成")
    print()
    print(f"  最终统计:")
    print(f"    train: {len(train)} 条 (用于训练)")
    print(f"    val:   {len(val)} 条 (用于早停判断)")
    print(f"    test:  {len(test)} 条 (用于最终评估)")
    print(f"    总计:  {len(records)} 条")
    print()
    print(f"  下一步: 上 GPU, 跑 03_train_m3gnet_tc.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
