# v2 项目进度备忘录 (PROGRESS.md)

> 最后更新: 2026-06-03 (数据准备完成)
> 项目: superconductor Tc prediction with M3GNet
> 路径: D:\superc_ml\m3gnet_upgrade\

---

## 🎯 一句话总结

v2 已经完成本地所有准备工作 (数据集 + 预处理). 下一步: **上 GPU 跑训练**.

---

## ✅ 已完成

1. **环境**: conda env `superc_v2` (Python 3.10.20 + PyTorch 2.2.0 + matgl 1.3.0)
2. **数据集**: 3DSC (Sommer et al. Sci Data 2023)
   - 来源: https://github.com/aimat-lab/3DSC
   - 本地路径: `data/3dsc/3DSC_repo/`
   - 主表: `superconductors_3D/data/final/MP/3DSC_MP.csv` (5773 条)
   - cif 文件夹: 同上 + `cifs/` (10904 个 cif)
3. **预处理**: 完成
   - 输出在 `data/processed_3dsc/`
   - `train.pkl`: 4618 条
   - `val.pkl`: 577 条
   - `test.pkl`: 578 条

## ⏳ 下一步 (按顺序)

1. **上 GPU 跑训练** (优云 RTX 3090, ~6-10 小时, ~¥15)
2. **评估** (CPU 即可)
3. **多形态对比** (验证结构敏感性)
4. **写 v2 报告**

---

## 📊 关键数据状态

### 家族分布 (按预处理结果)
| 家族 | 训练 | 验证 | 测试 | 总数 | max Tc (K) |
|---|---|---|---|---|---|
| BCS-Other | 2815 | 352 | 352 | 3519 | - |
| Cuprate | 707 | 88 | 88 | 883 | 132.9 |
| Hydride | 405 | 51 | 51 | 507 | 17.0 (常压) |
| Iron-based | 352 | 44 | 44 | 440 | 56.5 |
| A15 | 167 | 21 | 21 | 209 | 20.8 |
| MgB2-like | 97 | 12 | 12 | 121 | 41.4 |
| Nickelate | 75 | 9 | 10 | 94 | 92.5 |

### 重要发现
- **总样本数**: 5773 (完美, 0 失败)
- **Tc 上限**: 132.9 K (3DSC 不含 LaH10 等高压氢化物)
- **铜氧化物充足** (883 条, 主要训练目标)
- **镍基 94 条** (比预期多, 是个亮点)

---

## 🚀 明天上 GPU 的步骤

### 准备工作 (在本地)
```powershell
cd D:\superc_ml\m3gnet_upgrade
conda activate superc_v2

# 检查文件还在
ls data\processed_3dsc\
# 应该看到: train.pkl, val.pkl, test.pkl, stats.csv, families.csv
```

### 上优云的清单 (要上传的)
- ✅ 整个 `m3gnet_upgrade/` 文件夹 (代码 + 配置)
- ✅ `data/processed_3dsc/*.pkl` (8 MB, 处理好的数据)
- ❌ 不要上传 `data/3dsc/3DSC_repo/` (200 MB, 在云上重新下也行)
- ❌ 不要上传 `data/3dsc/3DSC_repo/.git/` (没用)

### 优云配置 (上次已确认)
- 镜像: **Miniconda3** (Ubuntu 22.04 + Python 3.x)
- GPU: **RTX 3090 (24G)** 或 **RTX 4090 (24G)**
- 实例类型: **独占式**
- 付款: **按量计费 ¥1.88/h** (调试期); 或包日 ¥41.48 (训练定型后)

### 云上要装的依赖
```bash
conda create -n superc_v2 python=3.10 -y
conda activate superc_v2
pip install -r requirements_v2.txt
# 验证: python -c "import torch; print(torch.cuda.is_available())"
# 应该输出 True
```

### 跑训练
```bash
python scripts/03_train_m3gnet_tc.py
# 后台跑:
# nohup python scripts/03_train_m3gnet_tc.py > train.log 2>&1 &
# tail -f train.log
```

---

## 🚨 已知问题 / 注意事项

### 1. `03_train_m3gnet_tc.py` 需要小改
当前训练脚本默认用 `data/processed/` 路径,
新数据在 `data/processed_3dsc/`. 
明天上 GPU 前修改 train_config.yaml 的路径:

```yaml
data:
  train_pkl: data/processed_3dsc/train.pkl
  val_pkl:   data/processed_3dsc/val.pkl
  test_pkl:  data/processed_3dsc/test.pkl
```

### 2. matgl 1.3.0 + PyTorch 2.2.0 + DGL 兼容性
本地装的版本组合是 OK 的, 但云上重装可能踩坑.
**应对**: 把本地 `requirements_v2.txt` 完全复制, 别让 pip 自动选版本.

### 3. GPU 显存
batch_size=32 需要 18GB+ VRAM, RTX 3090 24GB 够.
如果 OOM, 改 train_config.yaml 里 `batch_size: 16`.

---

## 📁 项目文件结构 (当前)

```
D:\superc_ml\
├── (v1 原文件保持不动) ✅
│
└── m3gnet_upgrade/                     ← v2 工作目录
    ├── PROGRESS.md                      ← 本文档
    ├── README.md
    ├── requirements_v2.txt
    ├── m3gnet_predictor.py
    ├── app_v2.py                        (Streamlit UI, 等训练完再用)
    │
    ├── configs/
    │   └── train_config.yaml            (训练超参数)
    │
    ├── scripts/
    │   ├── 00_polymorph_demo.py
    │   ├── 01_download_3dsc.py          ✅ 已运行
    │   ├── 01b_diagnose_3dsc.py         (诊断辅助, 可删)
    │   ├── 01c_read_3dsc_mp.py          (诊断辅助, 可删)
    │   ├── 02_prepare_dataset_3dsc.py   ✅ 已运行
    │   ├── 03_train_m3gnet_tc.py        ⏳ 下一步, GPU
    │   ├── 04_evaluate.py               ⏳ 训练后
    │   ├── 05_predict_polymorphs.py     ⏳
    │   ├── 06_compare_with_v1.py        ⏳
    │   └── 07_build_extension_test_set.py  (可选, 手动构造关键化合物)
    │
    ├── data/
    │   ├── 3dsc/3DSC_repo/              (200 MB, 原始数据)
    │   ├── processed_3dsc/               ⭐ 训练用数据
    │   │   ├── train.pkl                 (4618 条)
    │   │   ├── val.pkl                   (577 条)
    │   │   ├── test.pkl                  (578 条)
    │   │   ├── stats.csv
    │   │   └── families.csv
    │   └── extension_test/                (可选, 手动构造的 cif)
    │
    ├── models/                          (训练后存模型)
    └── results/                         (评估输出)
```

---

## 💰 预算情况

- 已花: ¥0
- 预算上限: ¥200
- 预计训练总花费: ¥30-50 (按 ¥1.88/h × 8h + 调试 2-3 次)
- 充足

---

## 🔗 重要链接

- 项目 GitHub: https://github.com/Kruger114514/superc_ml
- v1 demo: https://supercml-song.streamlit.app
- 3DSC paper: https://www.nature.com/articles/s41597-023-02721-y
- matgl docs: https://matgl.ai
- 优云控制台: https://console.compshare.cn

---

## 🤖 和 AI 协作的状态

如果明天回来继续, 把这个 PROGRESS.md 发给 AI 助手, 
它能立刻接手不用从头介绍.

关键决策记录:
- ❌ 不用 SuperBand (cif 数据不完整)
- ❌ 不用 JARVIS (Tc 是 DFT 计算, 不是实验)
- ✅ 用 3DSC (实验 Tc + 完整 cif)
- ❌ 不用 COD/拓扑数据库 (师兄推荐但和 Tc 任务不直接相关)
- ✅ 报告聚焦"方法学升级 (组分 -> 结构)", 不强求预测 LaH10
