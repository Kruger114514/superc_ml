"""
Baseline 模型: 用 81 个组分特征预测超导临界温度 Tc
对比 Random Forest 和 XGBoost
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os

# ===== 1. 加载数据 =====
print("=" * 50)
print("步骤 1: 加载数据")
print("=" * 50)
df = pd.read_csv('data/raw/train.csv')
print(f"数据形状: {df.shape}")
print(f"特征数: {df.shape[1] - 1}")
print(f"样本数: {df.shape[0]}")

# 分离特征 X 和目标 y
X = df.drop(columns=['critical_temp'])
y = df['critical_temp']

print(f"\nTc 统计信息:")
print(f"  均值: {y.mean():.2f} K")
print(f"  中位数: {y.median():.2f} K")
print(f"  最大值: {y.max():.2f} K")
print(f"  最小值: {y.min():.2f} K")

# ===== 2. 划分训练/测试集 =====
print("\n" + "=" * 50)
print("步骤 2: 划分训练集和测试集 (80/20)")
print("=" * 50)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集: {X_train.shape[0]} 条")
print(f"测试集: {X_test.shape[0]} 条")

# ===== 3. 训练 Random Forest =====
print("\n" + "=" * 50)
print("步骤 3: 训练 Random Forest")
print("=" * 50)
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,          # 用所有 CPU 核
    random_state=42
)
print("训练中... (约 30 秒-1 分钟)")
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest 结果:")
print(f"  MAE  = {rf_mae:.3f} K")
print(f"  R²   = {rf_r2:.4f}")

# ===== 4. 训练 XGBoost =====
print("\n" + "=" * 50)
print("步骤 4: 训练 XGBoost")
print("=" * 50)
xgb = XGBRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)
print("训练中... (约 30 秒-1 分钟)")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost 结果:")
print(f"  MAE  = {xgb_mae:.3f} K")
print(f"  R²   = {xgb_r2:.4f}")

# ===== 5. 可视化对比 =====
print("\n" + "=" * 50)
print("步骤 5: 生成预测对比图")
print("=" * 50)
os.makedirs('results', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 子图 1: Random Forest
axes[0].scatter(y_test, rf_pred, alpha=0.3, s=10)
axes[0].plot([0, 140], [0, 140], 'r--', label='perfect')
axes[0].set_xlabel('True Tc (K)')
axes[0].set_ylabel('Predicted Tc (K)')
axes[0].set_title(f'Random Forest\nMAE={rf_mae:.2f} K, R²={rf_r2:.3f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 子图 2: XGBoost
axes[1].scatter(y_test, xgb_pred, alpha=0.3, s=10, color='orange')
axes[1].plot([0, 140], [0, 140], 'r--', label='perfect')
axes[1].set_xlabel('True Tc (K)')
axes[1].set_ylabel('Predicted Tc (K)')
axes[1].set_title(f'XGBoost\nMAE={xgb_mae:.2f} K, R²={xgb_r2:.3f}')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
save_path = 'results/baseline_predictions.png'
plt.savefig(save_path, dpi=120, bbox_inches='tight')
print(f"图表已保存: {save_path}")

# ===== 6. 特征重要性 (XGBoost) =====
print("\n" + "=" * 50)
print("步骤 6: 最重要的 10 个特征 (XGBoost)")
print("=" * 50)
importance = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance.head(10).to_string())

# 保存特征重要性图
fig, ax = plt.subplots(figsize=(8, 6))
importance.head(15).plot(kind='barh', ax=ax)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance')
ax.set_title('Top 15 Important Features (XGBoost)')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=120, bbox_inches='tight')
print(f"特征重要性图已保存: results/feature_importance.png")

print("\n" + "=" * 50)
print("Baseline 完成!")
print("=" * 50)
print(f"\n模型对比:")
print(f"  Random Forest: MAE = {rf_mae:.2f} K, R² = {rf_r2:.3f}")
print(f"  XGBoost      : MAE = {xgb_mae:.2f} K, R² = {xgb_r2:.3f}")