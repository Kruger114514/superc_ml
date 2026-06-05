"""剔除 OOD outlier 后重算真实性能"""
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv('results/v2_test_predictions.csv')

# 剔除明显的 OOD outlier (abs_error > 100 K)
df_clean = df[df['abs_error'] < 100].copy()
df_outliers = df[df['abs_error'] >= 100].copy()

print("=" * 70)
print("  OOD Outlier 检测")
print("=" * 70)
print(f"  剔除 {len(df) - len(df_clean)} 个 OOD outlier")
print(f"  ({(len(df) - len(df_clean))/len(df)*100:.2f}% of test set)\n")
print("  Outlier 样本:")
print(df_outliers[['formula', 'family', 'true_tc', 'pred_tc', 'abs_error']].to_string(index=False))

print()
print("=" * 70)
print(f"  真实性能 (剔除 outlier 后, n={len(df_clean)}):")
print("=" * 70)
print(f"  R² = {r2_score(df_clean['true_tc'], df_clean['pred_tc']):.3f}")
print(f"  MAE = {mean_absolute_error(df_clean['true_tc'], df_clean['pred_tc']):.2f} K")

print()
print("=" * 70)
print("  Per-family (剔除 outlier 后):")
print("=" * 70)
for family, grp in df_clean.groupby('family'):
    n = len(grp)
    mae = mean_absolute_error(grp['true_tc'], grp['pred_tc'])
    if n > 1:
        r2 = r2_score(grp['true_tc'], grp['pred_tc'])
        print(f"  {family:15s}  n={n:4d}  MAE={mae:6.2f} K  R²={r2:6.3f}")
    else:
        print(f"  {family:15s}  n={n:4d}  MAE={mae:6.2f} K  R²= -")