"""
深度分析: 模型对不同超导体家族的预测表现
生成 5 张图 + 1 个统计表, 用于报告

家族分类:
  - BCS         传统电声耦合
  - Cuprates    铜氧化物
  - Iron-based  铁基
  - Hydride     高压氢化物 (训练集几乎没有)
  - Nickelate   镍基 (训练集没有)
  - Other       其他
"""
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from predict import TcPredictor

# ===== 配置 =====
os.makedirs('results/families', exist_ok=True)

# 让 matplotlib 显示英文标签 (LaTeX 报告里更通用)
plt.rcParams.update({
    'font.size': 11,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

FAMILY_COLORS = {
    'BCS':        '#3b82f6',  # blue
    'Cuprate':    '#10b981',  # green
    'Iron-based': '#f59e0b',  # amber
    'Hydride':    '#ef4444',  # red
    'Nickelate':  '#8b5cf6',  # purple
    'Other':      '#9ca3af',  # gray
}

# ===== 著名超导体清单 (从文献整理) =====
# 来源: Annu. Rev. Condens. Matter Phys. 综述 + Wikipedia
FAMOUS = pd.DataFrame([
    # === BCS 传统 ===
    ('Hg',         4.2,   'BCS'),
    ('Pb',         7.2,   'BCS'),
    ('Nb',         9.3,   'BCS'),
    ('Nb3Sn',      18.0,  'BCS'),
    ('Nb3Ge',      23.0,  'BCS'),
    ('MgB2',       39.0,  'BCS'),
    # === 铜氧化物 ===
    ('La2CuO4',    38.0,  'Cuprate'),     # 母体掺杂前
    ('YBa2Cu3O7',  92.0,  'Cuprate'),     # YBCO
    ('Bi2Sr2CaCu2O8', 95.0, 'Cuprate'),   # Bi-2212
    ('Bi2Sr2Ca2Cu3O10', 110.0, 'Cuprate'),# Bi-2223
    ('Tl2Ba2Ca2Cu3O10', 125.0, 'Cuprate'),
    ('HgBa2Ca2Cu3O8',  133.0, 'Cuprate'), # 常压最高 Tc
    # === 铁基 ===
    ('LaFeAsO',    26.0,  'Iron-based'),
    ('SmFeAsO',    55.0,  'Iron-based'),
    ('BaFe2As2',   38.0,  'Iron-based'),
    ('FeSe',       8.0,   'Iron-based'),
    ('FeSe0.5Te0.5', 14.0, 'Iron-based'),
    # === 高压氢化物 (压力下 Tc, 训练集没见过这种条件) ===
    ('H3S',        203.0, 'Hydride'),     # 155 GPa
    ('LaH10',      250.0, 'Hydride'),     # 170 GPa
    ('YH9',        243.0, 'Hydride'),     # 200 GPa
    ('YH6',        224.0, 'Hydride'),     # 166 GPa
    ('CaH6',       215.0, 'Hydride'),     # 170 GPa
    ('CeH9',       100.0, 'Hydride'),     # 100 GPa
    # === 镍基 (新兴, 训练集大概率没见过) ===
    ('Nd0.8Sr0.2NiO2', 9.0,  'Nickelate'),  # 第一个无限层镍基
    ('La3Ni2O7',   80.0,  'Nickelate'),     # 14 GPa 下
    ('Pr0.8Sr0.2NiO2', 12.0, 'Nickelate'),
], columns=['formula', 'exp_tc', 'family'])


# ===== 加载模型并预测 =====
print("=" * 60)
print("加载模型, 预测著名超导体的 Tc")
print("=" * 60)
predictor = TcPredictor()
print(f"模型测试集: R² = {predictor.test_r2:.3f}, MAE = {predictor.test_mae:.2f} K\n")

pred_results = []
for _, row in FAMOUS.iterrows():
    r = predictor.predict(row['formula'])
    if r['success']:
        pred_results.append({
            'formula': row['formula'],
            'exp_tc': row['exp_tc'],
            'pred_tc': r['predicted_tc'],
            'family': row['family'],
            'confidence': r['confidence'],
        })
    else:
        print(f"  ⚠️  {row['formula']}: 预测失败 - {r['error']}")

results_df = pd.DataFrame(pred_results)
results_df['error'] = results_df['pred_tc'] - results_df['exp_tc']
results_df['abs_error'] = results_df['error'].abs()
results_df['rel_error'] = results_df['abs_error'] / results_df['exp_tc']

# ===== 统计表 =====
print("=" * 60)
print("按家族汇总")
print("=" * 60)
stats = results_df.groupby('family').agg(
    n=('formula', 'count'),
    mean_exp_tc=('exp_tc', 'mean'),
    mean_pred_tc=('pred_tc', 'mean'),
    MAE=('abs_error', 'mean'),
    max_abs_err=('abs_error', 'max'),
).round(2)
print(stats.to_string())

stats.to_csv('results/families/family_stats.csv', encoding='utf-8-sig')
results_df.to_csv('results/families/all_predictions.csv', index=False, encoding='utf-8-sig')

# ===== 图 1: 预测 vs 实验散点图 (按家族着色) =====
print("\n生成图 1: 预测 vs 实验散点图...")
fig, ax = plt.subplots(figsize=(8, 8))
for family, group in results_df.groupby('family'):
    ax.scatter(group['exp_tc'], group['pred_tc'],
               color=FAMILY_COLORS.get(family, '#888'),
               s=90, alpha=0.85, edgecolor='white', linewidth=1.2,
               label=f'{family} (n={len(group)})')

max_val = max(results_df['exp_tc'].max(), results_df['pred_tc'].max()) + 20
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Perfect prediction')
ax.fill_between([0, max_val], [0, max_val * 0.7], [0, max_val * 1.3],
                alpha=0.1, color='gray', label='±30% band')

ax.set_xlabel('Experimental $T_c$ (K)', fontsize=13)
ax.set_ylabel('Predicted $T_c$ (K)', fontsize=13)
ax.set_title('Model performance across superconductor families', fontsize=14)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3)
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_aspect('equal')
plt.savefig('results/families/fig1_pred_vs_exp.png')
plt.close()

# ===== 图 2: 按家族的误差分布 (箱线图) =====
print("生成图 2: 家族误差箱线图...")
fig, ax = plt.subplots(figsize=(10, 6))
families_ordered = ['BCS', 'Cuprate', 'Iron-based', 'Hydride', 'Nickelate']
families_ordered = [f for f in families_ordered if f in results_df['family'].values]

box_data = [results_df[results_df['family'] == f]['error'].values for f in families_ordered]
bp = ax.boxplot(box_data, labels=families_ordered, patch_artist=True, widths=0.55)
for patch, family in zip(bp['boxes'], families_ordered):
    patch.set_facecolor(FAMILY_COLORS.get(family, '#888'))
    patch.set_alpha(0.7)

# 叠加散点
for i, family in enumerate(families_ordered):
    errors = results_df[results_df['family'] == family]['error'].values
    x = np.random.normal(i + 1, 0.04, size=len(errors))
    ax.scatter(x, errors, color=FAMILY_COLORS[family], s=40,
               edgecolor='black', linewidth=0.5, alpha=0.9, zorder=3)

ax.axhline(0, color='red', linestyle='--', alpha=0.6, label='Zero error')
ax.set_ylabel('Prediction error: Predicted − Experimental (K)', fontsize=12)
ax.set_title('Prediction error by superconductor family', fontsize=14)
ax.grid(alpha=0.3, axis='y')
ax.legend()
plt.savefig('results/families/fig2_error_by_family.png')
plt.close()

# ===== 图 3: 训练数据 Tc 分布 + 五个家族实验位置 =====
print("生成图 3: 训练 Tc 分布 + 家族实验位置...")
train_df = pd.read_csv('data/raw/unique_m.csv')
fig, ax = plt.subplots(figsize=(13, 5))
ax.hist(train_df['critical_temp'], bins=100, alpha=0.5,
        color='steelblue', label=f'Training data (n={len(train_df)})')

for family in families_ordered:
    tcs = results_df[results_df['family'] == family]['exp_tc'].values
    ax.scatter(tcs, np.full_like(tcs, 3 + families_ordered.index(family) * 50),
               color=FAMILY_COLORS[family], s=100, edgecolor='black',
               linewidth=1, alpha=0.95, label=family, zorder=5)

ax.set_xlabel('Critical temperature $T_c$ (K)', fontsize=12)
ax.set_ylabel('Count (histogram) / arbitrary y (markers)', fontsize=12)
ax.set_title('Training data $T_c$ distribution vs. famous superconductors', fontsize=14)
ax.legend(loc='upper right', ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim(-5, 280)
plt.savefig('results/families/fig3_tc_distribution.png')
plt.close()

# ===== 图 4: 氢化物专项失败图 (柱状对比) =====
print("生成图 4: 氢化物专项失败...")
hydrides = results_df[results_df['family'] == 'Hydride'].sort_values('exp_tc')
if len(hydrides) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(hydrides))
    width = 0.38
    ax.bar(x - width/2, hydrides['exp_tc'], width,
           color='#ef4444', alpha=0.85, label='Experimental', edgecolor='black')
    ax.bar(x + width/2, hydrides['pred_tc'], width,
           color='#fca5a5', alpha=0.85, label='Predicted', edgecolor='black', hatch='//')

    # 标误差
    for i, (_, row) in enumerate(hydrides.iterrows()):
        ax.annotate(f'Δ={row["error"]:+.0f}K',
                    xy=(i, max(row['exp_tc'], row['pred_tc']) + 10),
                    ha='center', fontsize=9, color='darkred')

    ax.set_xticks(x)
    ax.set_xticklabels(hydrides['formula'], rotation=15)
    ax.set_ylabel('$T_c$ (K)', fontsize=12)
    ax.set_title('High-pressure hydrides: model fails on the extrapolation region',
                 fontsize=13)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, axis='y')
    plt.savefig('results/families/fig4_hydride_failure.png')
    plt.close()

# ===== 图 5: PCA 投影, 看测试家族在训练分布的位置 =====
print("生成图 5: PCA 特征空间投影...")
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

ep = ElementProperty.from_preset("magpie")

# 用一个训练集子样本 (3000 条) 节省时间
sample_train = train_df.sample(n=min(3000, len(train_df)), random_state=42).reset_index(drop=True)
print(f"  对 {len(sample_train)} 个训练样本提取特征 (约 1-2 分钟)...")

train_feats = []
for f in sample_train['material']:
    try:
        train_feats.append(ep.featurize(Composition(f)))
    except Exception:
        train_feats.append([np.nan] * len(ep.feature_labels()))
train_feats = np.array(train_feats)
mask = ~np.isnan(train_feats).any(axis=1)
train_feats = train_feats[mask]
train_tcs = sample_train.loc[mask, 'critical_temp'].values

# 测试家族特征
test_feats = []
test_families = []
test_formulas = []
for _, row in results_df.iterrows():
    try:
        test_feats.append(ep.featurize(Composition(row['formula'])))
        test_families.append(row['family'])
        test_formulas.append(row['formula'])
    except Exception:
        pass
test_feats = np.array(test_feats)

# PCA 在合并空间上拟合, 投影到 2D
all_feats = np.vstack([train_feats, test_feats])
pca = PCA(n_components=2)
all_2d = pca.fit_transform(all_feats)
train_2d = all_2d[:len(train_feats)]
test_2d = all_2d[len(train_feats):]

fig, ax = plt.subplots(figsize=(10, 8))
# 训练背景: 用 Tc 上色
sc = ax.scatter(train_2d[:, 0], train_2d[:, 1], c=train_tcs,
                cmap='viridis', s=8, alpha=0.4)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Training $T_c$ (K)')

# 测试家族
for family in families_ordered:
    mask_f = np.array([f == family for f in test_families])
    if mask_f.any():
        ax.scatter(test_2d[mask_f, 0], test_2d[mask_f, 1],
                   color=FAMILY_COLORS[family], s=130, edgecolor='black',
                   linewidth=1.2, label=family, zorder=5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
ax.set_title('Compositional feature space: training data vs. test families',
             fontsize=13)
ax.legend(loc='best')
ax.grid(alpha=0.3)
plt.savefig('results/families/fig5_pca_projection.png')
plt.close()

# ===== 总结打印 =====
print("\n" + "=" * 60)
print("✅ 完成! 文件输出到 results/families/")
print("=" * 60)
print("\n生成的文件:")
print("  - fig1_pred_vs_exp.png        散点图: 预测 vs 实验, 按家族")
print("  - fig2_error_by_family.png    箱线图: 误差按家族")
print("  - fig3_tc_distribution.png    训练分布 + 家族位置")
print("  - fig4_hydride_failure.png    氢化物失败案例")
print("  - fig5_pca_projection.png     PCA 特征空间")
print("  - family_stats.csv            按家族统计表")
print("  - all_predictions.csv         全部预测原始数据")
print("\n下一步: 把这些发给我, 我会基于真实数据写报告大纲。")