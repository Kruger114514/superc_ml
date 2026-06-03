"""
02_prepare_dataset.py — 准备训练数据
====================================

把 3DSC 的 cif 文件 + Tc 标签转换为 M3GNet 需要的格式:
1. 读取 SC_MP.csv (formula, Tc, MP-id 列)
2. 加载对应的 cif 文件 → pymatgen.Structure 对象
3. 数据清洗: 去掉 Tc=0 或无效结构的条目
4. 按家族分层划分 train/val/test (8/1/1)
5. 保存为 pickle, 供训练脚本快速加载

运行: python scripts/02_prepare_dataset.py
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")

# 路径
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "3dsc"
CIFS_DIR = DATA_DIR / "3dsc_repo" / "superconductors_3D" / "data" / "final" / "MP" / "cifs"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "superconductors_3DSC.csv"

# 简易家族标签器 (复用 v1 analyze_families.py 的逻辑, 在分层划分时用)
def family_of(formula: str) -> str:
    f = formula.lower()
    if "h" in f and any(e in f for e in ["la", "y", "ce", "ca", "h3s", "lah"]):
        return "Hydride"
    if "cu" in f and "o" in f:
        return "Cuprate"
    if "fe" in f and ("as" in f or "se" in f or "te" in f or "p" in f):
        return "Iron-based"
    if "ni" in f and "o" in f:
        return "Nickelate"
    return "Other"  # 包括 BCS, intermetallic, etc.


def load_3dsc_csv(csv_path: Path) -> pd.DataFrame:
    """读取 3DSC 主表"""
    print(f"[1/5] 读取 {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"      原始记录数: {len(df)}")
    print(f"      列名: {list(df.columns)}")

    # 3DSC 的标准列名 (根据 Nature SciData 论文):
    # formula, tc, mp_id, source
    # 实际列名可能略不同, 按需调整
    expected_cols = {"formula", "tc"}
    available = set(df.columns.str.lower())
    if not expected_cols.issubset(available):
        # 尝试通用列名
        rename_map = {}
        for col in df.columns:
            if col.lower() in {"formula", "material", "name"}:
                rename_map[col] = "formula"
            elif col.lower() in {"tc", "critical_temp", "tc_k"}:
                rename_map[col] = "tc"
            elif col.lower() in {"mp_id", "mp-id", "mpid", "materials_id"}:
                rename_map[col] = "mp_id"
        df = df.rename(columns=rename_map)
        print(f"      重命名后列名: {list(df.columns)}")

    return df


def load_structure(mp_id: str, formula: str) -> Structure | None:
    """从 cif 加载结构。失败返回 None"""
    # 3DSC 的 cif 文件名约定: 通常是 <mp_id>.cif 或 <formula>.cif
    candidates = [
        CIFS_DIR / f"{mp_id}.cif",
        CIFS_DIR / f"{formula}.cif",
        CIFS_DIR / f"{mp_id}_{formula}.cif",
    ]
    for path in candidates:
        if path.exists():
            try:
                return Structure.from_file(str(path))
            except Exception as e:
                print(f"      警告: {path.name} 读取失败: {e}")
                return None
    return None


def process_dataset(df: pd.DataFrame) -> Tuple[List[Dict], pd.DataFrame]:
    """把每一行变成 {structure, tc, formula, family} 的 dict"""
    print(f"[2/5] 加载晶体结构 (从 {CIFS_DIR})...")

    records = []
    failed = 0
    for idx, row in df.iterrows():
        formula = str(row.get("formula", "")).strip()
        tc = row.get("tc", np.nan)
        mp_id = str(row.get("mp_id", "")).strip() if "mp_id" in df else ""

        # 数据清洗: Tc 必须正且有效
        if not isinstance(tc, (int, float)) or np.isnan(tc) or tc <= 0:
            failed += 1
            continue

        struct = load_structure(mp_id, formula)
        if struct is None:
            failed += 1
            continue

        records.append({
            "formula": formula,
            "tc": float(tc),
            "mp_id": mp_id,
            "family": family_of(formula),
            "structure": struct,
            "n_atoms": len(struct),
            "volume": float(struct.volume),
        })

        if (idx + 1) % 500 == 0:
            print(f"      已处理 {idx + 1}/{len(df)}, 成功 {len(records)}, 失败 {failed}")

    print(f"      ✓ 成功加载 {len(records)} 条 (失败 {failed})")

    stats_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "structure"}
        for r in records
    ])
    return records, stats_df


def stratified_split(
    records: List[Dict],
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """按家族分层划分 train/val/test (8/1/1)"""
    print(f"[3/5] 分层划分: train/val/test = "
          f"{1-test_size-val_size:.0%}/{val_size:.0%}/{test_size:.0%}")

    families = [r["family"] for r in records]

    # 先分出 test
    train_val, test = train_test_split(
        records, test_size=test_size, stratify=families, random_state=random_state,
    )

    # 再从 train_val 分出 val
    train_val_families = [r["family"] for r in train_val]
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val_families,
        random_state=random_state,
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        sizes = pd.Series([r["family"] for r in split]).value_counts().to_dict()
        print(f"      {name:5s}: {len(split):4d} 条, 家族分布 {sizes}")

    return train, val, test


def save(splits: Dict[str, List[Dict]], stats_df: pd.DataFrame):
    """保存 pickle + stats csv"""
    print(f"[4/5] 保存到 {PROCESSED_DIR}")

    for name, data in splits.items():
        out = PROCESSED_DIR / f"{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(data, f)
        print(f"      ✓ {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")

    stats_df.to_csv(PROCESSED_DIR / "dataset_stats.csv", index=False)
    print(f"      ✓ dataset_stats.csv")


def diagnose_hydrides(stats_df: pd.DataFrame):
    """检查 3DSC 里有没有高压氢化物 (LaH10, H3S, YH9 等)"""
    print(f"[5/5] 诊断: 检查 3DSC 数据集里是否包含高压氢化物")

    # 关键化合物 (从我们 v1 用的 26 个测试集里的氢化物)
    targets = ["LaH10", "YH6", "YH9", "CeH9", "H3S", "CaH6"]
    formulas = stats_df["formula"].astype(str).tolist()

    found = {}
    for t in targets:
        matches = [f for f in formulas if t.lower() in f.lower()]
        found[t] = matches

    print()
    for t, matches in found.items():
        if matches:
            print(f"      ✓ {t}: 找到 {len(matches)} 条, 例如 {matches[:3]}")
        else:
            print(f"      ✗ {t}: 未找到 (确认: 3DSC 不含此化合物)")
    print()
    n_found = sum(1 for m in found.values() if m)
    print(f"      高压氢化物覆盖: {n_found}/{len(targets)}")
    if n_found < len(targets) / 2:
        print(f"      ⚠ 大部分氢化物不在 3DSC 中")
        print(f"      预期: v2 在氢化物上仍可能失败, 但原因从 'Magpie 不含压力' 变为 '3DSC 缺数据'")
    print()


def main():
    print("=" * 70)
    print("  Prepare Dataset for M3GNet Training")
    print("=" * 70)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {CSV_PATH}. 请先跑 01_download_3dsc.py"
        )
    if not CIFS_DIR.exists():
        raise FileNotFoundError(
            f"找不到 cif 文件夹 {CIFS_DIR}. "
            f"请按 01_download_3dsc.py 提示克隆 3DSC 仓库."
        )

    df = load_3dsc_csv(CSV_PATH)
    records, stats_df = process_dataset(df)
    if len(records) == 0:
        raise RuntimeError("没有成功加载任何记录, 请检查数据集路径和列名")

    train, val, test = stratified_split(records)
    save({"train": train, "val": val, "test": test}, stats_df)
    diagnose_hydrides(stats_df)

    print("=" * 70)
    print("  数据准备完成. 下一步: python scripts/03_train_m3gnet_tc.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
