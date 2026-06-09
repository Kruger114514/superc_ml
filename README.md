# 超导临界温度 Tc 预测器  ·  Superconductor Tc Prediction

> 基于机器学习的超导临界温度预测项目。课程大作业 (v1) + 方法学升级 (v2)。

[![Streamlit App](https://img.shields.io/badge/Streamlit-online-brightgreen)](https://supercml-song.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)](https://pytorch.org/)

---

## 📌 项目演进

| 版本 | 方法 | 数据 | Test MAE | 状态 |
|---|---|---|---|---|
| **v1** | XGBoost + Magpie (132-d 组分特征) | SuperCon (17,010) | 5.48 K | ✅ 已部署 |
| **v2** | **M3GNet** (图神经网络, 结构感知) | **3DSC** (5,773, 含 cif 结构) | **4.40 K** | ✅ 训练完成 |

**v2 核心结论**: 用 **3.7× 更少**的训练数据, 实现 **20% 更低**的测试 MAE。

---

## 🌐 在线演示 (v1)

🔗 **https://supercml-song.streamlit.app/**

输入化学式 (如 `MgB2`, `YBa2Cu3O7`, `LaH10`) 即可获得 Tc 预测。

---

## 🎯 v2 关键结果

### 整体性能
- **Val MAE**: 3.29 K (best @ epoch 168, early stop)
- **Test MAE**: 4.40 K  (after removing 1 OOD outlier)
- **Test R²**: 0.786

### 按家族分解
| Family | n | MAE (K) | R² |
|---|---|---|---|
| Hydride        | 51  | **1.04** | 0.65 |
| BCS-Other      | 351 | 1.63 | 0.62 |
| A15            | 21  | 1.83 | 0.86 |
| **Nickelate**  | 10  | 3.14 | **0.96** ⭐ |
| MgB2-like      | 12  | 6.25 | -0.88 |
| Iron-based     | 44  | 9.50 | 0.10 |
| Cuprate        | 88  | 15.30 | 0.71 |

**4/7 家族 (Hydride / BCS / A15 / Nickelate) 共 433 样本 (75%) MAE < 4 K**，全部优于 v1 baseline。

---

## 🔬 算法与技术栈

### v1 (Baseline)

```
化学式 → Magpie 特征 (132-d) → XGBoost → Tc 预测
```

| 组件 | 技术 |
|---|---|
| 特征工程 | Magpie elemental descriptors |
| 回归模型 | XGBoost (gradient-boosted trees) |
| 超参调优 | RandomizedSearchCV |
| 部署 | Streamlit |

### v2 (Structure-Aware)

```
cif 结构 → 原子图 (DGL) → M3GNet (预训练 + 微调) → log(Tc+1) → Tc 预测
```

| 组件 | 技术 |
|---|---|
| 结构读取 | `pymatgen` |
| 图构造 | `DGL` (5 Å 截断半径) |
| 核心模型 | `M3GNet` (Chen & Ong, Nat. Comp. Sci. 2022) |
| 预训练任务 | Formation Energy (Materials Project, 1.86M 样本) |
| 微调任务 | Tc 回归 (3DSC, 4618 训练样本) |
| 训练框架 | `PyTorch Lightning` + `matgl` |
| 优化器 | AdamW + Cosine Annealing |
| 训练硬件 | NVIDIA RTX 4090D (AutoDL 云租) |
| 训练时长 | 50 分钟, 200 epoch (early stop at 198) |

---

## 📁 项目结构

```
superc_ml/
│
├── 🌟 v1 (main 分支)
│   ├── app.py                          # Streamlit Web UI
│   ├── predict.py                      # 预测核心逻辑
│   ├── train_model.py                  # 模型训练脚本
│   ├── baseline.py                     # 基线模型对比
│   ├── download_data.py                # SuperCon 数据下载
│   ├── data/raw/                       # SuperCon 数据
│   └── models/tc_predictor.pkl         # 训练好的 XGBoost
│
└── 🚀 v2 (m3gnet-upgrade 分支)
    └── m3gnet_upgrade/
        ├── models/m3gnet_tc.pt          # 训练好的 M3GNet (3.5 MB)
        ├── scripts/
        │   ├── 01_download_3dsc.py      # 下载 3DSC
        │   ├── 02_prepare_dataset.py    # 数据预处理 & 分层划分
        │   ├── 03_train_m3gnet_tc.py    # 训练 (GPU)
        │   └── 04_evaluate.py           # 评估 (CPU)
        ├── evaluate_clean.py            # OOD outlier 剔除 & 重算
        ├── plot_v2_clean_figures.py     # 主要结果 figures
        ├── plot_v2_training_curves.py   # 训练曲线
        └── results/
            ├── figures/                  # 9 张 figures
            │   ├── v2_mae_by_family.png ⭐ 核心图
            │   ├── v2_vs_v1_mae.png     ⭐ v1 vs v2
            │   ├── v2_pred_vs_exp_clean.png
            │   └── v2_per_family_comparison.png
            ├── v2_test_predictions.csv  # 578 个测试预测
            ├── v2_per_family_metrics.csv
            └── v2_final_metrics.txt     # 数字摘要
```

---

## 🚀 本地运行

### v1: 启动 Streamlit Web 应用

```bash
git checkout main
pip install -r requirements.txt
streamlit run app.py
```

### v2: 训练 / 评估 M3GNet

```bash
git checkout m3gnet-upgrade
cd m3gnet_upgrade

# 安装依赖 (要 Python 3.10, CUDA 11.8 或 CPU)
pip install -r requirements_v2.txt
# 关键依赖: torch==2.2.0, matgl==1.3.0, dgl==1.1.3, numpy<2

# 1. 下载 3DSC 数据
python scripts/01_download_3dsc.py

# 2. 预处理 & 分层划分
python scripts/02_prepare_dataset.py

# 3. 训练 (推荐 GPU, ~50 分钟 on RTX 4090)
python scripts/03_train_m3gnet_tc.py

# 4. 评估 (CPU 也行, ~30 分钟)
python scripts/04_evaluate.py

# 5. OOD outlier 剔除 + 重算指标
python evaluate_clean.py

# 6. 生成所有 figures
python plot_v2_clean_figures.py
python plot_v2_training_curves.py
```

---

## 📊 关键发现

### ✅ v2 的优势
1. **多形态感知**: 能区分同组分但不同晶体结构的化合物 (v1 做不到)
2. **新家族迁移**: 在 Nickelate (2019 年新发现) 上 R² = 0.96, 无需重新训练
3. **数据效率高**: 3.7× 更少数据达到更好性能, 得益于预训练
4. **端到端学习**: 不依赖手工特征, 直接从 cif 学

### ⚠️ v2 的局限
1. **Cuprate 高 Tc 压缩**: 80-130 K 范围预测系统性偏低
2. **Iron-based 难捕捉**: R² 仅 0.10, 配对机制 (自旋涨落) 超出 GNN 预训练任务
3. **MgB2 mode collapse**: 12 个样本预测都聚集在 ~32 K
4. **分子晶体 OOD**: Ba₆C₆₀ (富勒烯) 预测出荒谬的 17,514 K, 揭示训练分布外的风险

### 🔬 一个有意思的发现
我们在测试集发现 1 个明显的 OOD outlier:

```
Ba6C60 (钡掺杂富勒烯, 实际 Tc = 7 K)
v2 预测: 17,514 K (比太阳表面还热)
```

剔除后整体 R² 从荒谬的 -1331 恢复到 0.786。**这个案例展示了 ML 模型在分布外样本上的失控风险**。

---

## 🔮 未来方向 (v3?)

- 家族专属 fine-tuning (Cuprate / Iron-based 子模型)
- 加入不确定性估计 (MC dropout / Deep Ensemble)
- 扩展到高压氢化物 (LaH₁₀ 类, Tc > 200 K)
- 集成 v1 的 LLM Agent → 对话式预测系统
- 与 DFT 电声耦合计算配合验证

---

## 📚 参考文献

- **3DSC**: Sommer, T. et al. *Sci Data* **10**, 816 (2023). [3DSC: a 3D superconductor materials database]
- **M3GNet**: Chen, C. & Ong, S. P. *Nat. Comput. Sci.* **2**, 718–728 (2022). [A universal graph deep learning interatomic potential for the periodic table]
- **SuperCon**: NIMS Japan, [https://supercon.nims.go.jp](https://supercon.nims.go.jp)
- **Magpie**: Ward, L. et al. *npj Comput. Mater.* **2**, 16028 (2016).
- **XGBoost**: Chen, T. & Guestrin, C. *KDD* (2016).

---

## 📝 引用

如果这个项目对你的研究有帮助:

```bibtex
@misc{song2026supercml,
  author = {Song, Ao},
  title  = {Superconductor Tc Prediction: From Composition (XGBoost) to Structure (M3GNet)},
  year   = {2026},
  url    = {https://github.com/Kruger114514/superc_ml}
}
```

---

## 🤝 致谢

- 感谢 Chen & Ong 团队开源 M3GNet
- 感谢 Sommer et al. 整理 3DSC 数据集
- 感谢 NIMS 维护 SuperCon 数据库

---

## 📧 联系方式

**作者**: Song Ao
**Email**: songao2025@shanghaitech.edu.cn
**GitHub**: [@Kruger114514](https://github.com/Kruger114514)

---

> *"From Composition to Structure — a methodological upgrade for superconductor Tc prediction."*
