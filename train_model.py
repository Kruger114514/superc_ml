"""
训练一个吃化学式的 Tc 预测模型
流程: 化学式 → matminer 提取 Magpie 特征 → XGBoost 训练 → 保存模型
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ===== 1. 加载化学式 + Tc =====
print("=" * 60)
print("步骤 1: 加载 unique_m.csv (化学式 + Tc)")
print("=" * 60)
df = pd.read_csv('data/raw/unique_m.csv')
print(f"原始数据: {df.shape[0]} 条")
print(f"列名: {df.columns.tolist()}")

# unique_m.csv 里有元素列 + critical_temp + material(化学式)
# 我们只需要 material 和 critical_temp 这两列
df = df[['material', 'critical_temp']].copy()
df = df.dropna()
print(f"去重前: {df.shape[0]} 条")
df = df.drop_duplicates(subset=['material'])
print(f"按化学式去重后: {df.shape[0]} 条")

# ===== 2. 化学式转 Composition 对象 =====
print("\n" + "=" * 60)
print("步骤 2: 解析化学式 (用 pymatgen)")
print("=" * 60)

def safe_composition(formula):
    """安全地把化学式转成 Composition 对象,失败返回 None"""
    try:
        return Composition(formula)
    except Exception:
        return None

tqdm.pandas(desc="解析化学式")
df['composition'] = df['material'].progress_apply(safe_composition)
n_before = len(df)
df = df.dropna(subset=['composition'])
print(f"成功解析: {len(df)}/{n_before} 条")

# ===== 3. 用 matminer 提取 Magpie 特征 =====
print("\n" + "=" * 60)
print("步骤 3: matminer 提取 Magpie 特征 (约 1-3 分钟)")
print("=" * 60)
print("Magpie 特征: 每个化学式生成 132 个数值特征")
print("(基于各元素的原子半径、电负性、价电子数等的统计量)")

ep = ElementProperty.from_preset("magpie")
# 顺序处理(避免多进程的坑),进度条可见
features_list = []
failed_idx = []
for idx, comp in enumerate(tqdm(df['composition'].tolist(), desc="特征提取")):
    try:
        feats = ep.featurize(comp)
        features_list.append(feats)
    except Exception:
        features_list.append([np.nan] * len(ep.feature_labels()))
        failed_idx.append(idx)

feature_names = ep.feature_labels()
X = pd.DataFrame(features_list, columns=feature_names, index=df.index)
print(f"特征矩阵形状: {X.shape}")
print(f"失败数: {len(failed_idx)}")

# 清理 NaN 行
mask = ~X.isna().any(axis=1)
X = X[mask].reset_index(drop=True)
y = df.loc[mask, 'critical_temp'].reset_index(drop=True)
formulas = df.loc[mask, 'material'].reset_index(drop=True)
print(f"清理 NaN 后: {len(X)} 条")

# ===== 4. 训练集/测试集 =====
print("\n" + "=" * 60)
print("步骤 4: 划分训练集和测试集")
print("=" * 60)
X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(
    X, y, formulas, test_size=0.2, random_state=42
)
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ===== 5. 训练 XGBoost =====
print("\n" + "=" * 60)
print("步骤 5: 训练 XGBoost")
print("=" * 60)
model = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)
print("训练中...")
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
train_r2 = r2_score(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"\n训练集: MAE = {train_mae:.2f} K, R² = {train_r2:.4f}")
print(f"测试集: MAE = {test_mae:.2f} K, R² = {test_r2:.4f}")

# ===== 6. 在几个有名的氢化物上测试 =====
print("\n" + "=" * 60)
print("步骤 6: 在著名超导体上验证")
print("=" * 60)

famous = [
    ("MgB2",       39,   "硼化镁 (BCS)"),
    ("YBa2Cu3O7",  92,   "YBCO (铜氧化物)"),
    ("FeSe",       8,    "铁基"),
    ("Nb3Sn",      18,   "传统超导体"),
    ("H3S",        203,  "高压氢化物"),
    ("LaH10",      250,  "高压氢化物"),
]

print(f"{'化学式':<12} {'实验 Tc':<10} {'预测 Tc':<10} {'误差':<10} 说明")
print("-" * 70)
for formula, exp_tc, note in famous:
    try:
        comp = Composition(formula)
        feats = np.array(ep.featurize(comp)).reshape(1, -1)
        pred = model.predict(feats)[0]
        err = pred - exp_tc
        print(f"{formula:<12} {exp_tc:<10} {pred:<10.2f} {err:<+10.2f} {note}")
    except Exception as e:
        print(f"{formula:<12} 错误: {e}")

# ===== 7. 保存模型和特征化器 =====
print("\n" + "=" * 60)
print("步骤 7: 保存模型")
print("=" * 60)
os.makedirs('models', exist_ok=True)

artifact = {
    'model': model,
    'feature_names': feature_names,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'n_train': len(X_train),
    'n_test': len(X_test),
}
save_path = 'models/tc_predictor.pkl'
joblib.dump(artifact, save_path)
print(f"模型已保存: {save_path}")
print(f"文件大小: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

print("\n" + "=" * 60)
print("✅ 训练完成！下一步: 写 Streamlit UI")
print("=" * 60)