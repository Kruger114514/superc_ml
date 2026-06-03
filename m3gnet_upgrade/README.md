# M3GNet Upgrade (v2) — 结构敏感的超导 Tc 预测

> v1 用化学组分预测 Tc (XGBoost + Magpie)。
> v2 用**晶体结构**预测 Tc (M3GNet + 3DSC)。
> 解决老师提出的问题：同组分不同结构（石墨 vs 金刚石）能否区分？

---

## 这个文件夹做什么

补全 v1 的根本局限：**让模型看到结构**。

v1 的输入是化学式，所以石墨和金刚石输入完全相同 → 预测完全相同。
v2 的输入是晶格 + 原子坐标，每个多形态都有独立的预测。

---

## 文件结构

```
m3gnet_upgrade/
├── README.md                       <- 本文档
├── requirements_v2.txt             <- v2 专用依赖 (PyTorch + matgl)
├── configs/
│   └── train_config.yaml           <- 训练超参数
├── scripts/
│   ├── 00_polymorph_demo.py        <- 最小演示: 用预训练模型对比石墨 vs 金刚石
│   ├── 01_download_3dsc.py         <- 下载 3DSC 数据集
│   ├── 02_prepare_dataset.py       <- 把 cif 转成 M3GNet 输入格式
│   ├── 03_train_m3gnet_tc.py       <- 训练主脚本 (需要 GPU)
│   ├── 04_evaluate.py              <- 在测试集上评估, 生成报告图
│   ├── 05_predict_polymorphs.py    <- 多形态对比表 (核心 figure)
│   └── 06_compare_with_v1.py       <- v1 (XGBoost) vs v2 (M3GNet) 对比
├── m3gnet_predictor.py             <- 封装好的 predict 类, 供 Streamlit 用
├── app_v2.py                       <- 升级版 Streamlit UI (3 个 tab)
├── models/
│   └── m3gnet_tc.pt                <- 训练好的模型权重 (训练后生成)
├── data/
│   └── 3dsc/                       <- 3DSC 数据集 (下载后填充)
└── results/
    ├── figures/                    <- 评估生成的图
    └── polymorph_comparison.csv    <- 多形态对比结果
```

---

## 快速开始 (4 个阶段)

### Stage 1: 不需要 GPU, 今天就能跑

```bash
# 装 v2 依赖 (建议新建 conda env)
conda create -n superc_v2 python=3.10
conda activate superc_v2
pip install -r requirements_v2.txt

# 跑最小演示, 验证 M3GNet 工作正常 + 给老师看
python scripts/00_polymorph_demo.py
```

预期输出: 石墨和金刚石的能量值不同，证明"结构进入模型后是可区分的"。

### Stage 2: 准备数据 (CPU 即可)

```bash
python scripts/01_download_3dsc.py     # 下载 3DSC
python scripts/02_prepare_dataset.py   # 转换格式 + 训练集/测试集划分
```

### Stage 3: 训练 (需要 GPU, 建议 24GB+)

```bash
python scripts/03_train_m3gnet_tc.py --config configs/train_config.yaml
# 训练时间: RTX 4090 约 4-8 小时, RTX 3090 约 8-12 小时
```

### Stage 4: 评估 + Web 部署 (CPU 即可)

```bash
python scripts/04_evaluate.py                 # 测试集性能
python scripts/05_predict_polymorphs.py       # 核心 figure
python scripts/06_compare_with_v1.py          # v1 vs v2 对比表

streamlit run app_v2.py                       # 启动新 UI
```

---

## v1 vs v2 关系

**v1 不被删除！** 这是关键。

- v1 (XGBoost) 仍在 `D:\superc_ml\` 主目录, 仍然部署在 Streamlit Cloud
- v2 在子文件夹 `m3gnet_upgrade/`, 独立 conda env, 不冲突
- 论文里**两者都报告**, 形成对比, 突出 v2 的优势

---

## 关键设计决策

1. **数据集用 3DSC-MP 而不是 3DSC-ICSD**:
   3DSC 论文显示 MP 子集噪声小, 适合做 baseline
2. **模型架构: 用 matgl 的 M3GNet** (PyTorch + DGL/PyG, 当前活跃维护)
   - 不用原版 m3gnet (TensorFlow, 已 archived)
3. **损失函数: MSLE (Mean Squared Log Error)**
   - 跟 3DSC 论文一致, 因为 Tc 跨好几个量级
4. **预训练策略: 从 `M3GNet-MP-2018.6.1-Eform` 开始 fine-tune**
   - 不从头训练 (小数据集 + 大模型 = 过拟合风险)
5. **架构选择: M3GNet 是 invariant GNN, 处理化学组分 + 结构**
   - 不用 equivariant 模型, Tc 是标量, invariant 已经够

---

## 已知局限 (报告里要写)

1. **3DSC 不包含高压氢化物的高压相结构**
   - LaH10 在 170 GPa 下的 Fm-3m 结构 MP 里没有
   - 因此 v2 在氢化物上**可能仍然失败**, 但失败原因变了:
     - v1: "特征不够" (Magpie 不含压力)
     - v2: "数据集没有" (3DSC 不含高压相)
   - 这是一个**真实的、值得讨论的发现**

2. **3DSC 的结构是"近似匹配"**
   - 3DSC 用启发式方法把 SuperCon 化学式匹配到 MP 结构
   - 部分匹配可能不是真实的实验结构
   - 数据集论文承认这点 (论文 §3.2 "limitations")

3. **训练需要 GPU**
   - 我们 v1 (XGBoost) 笔记本 CPU 就能跑
   - v2 必须 GPU, 这是工程上的代价
