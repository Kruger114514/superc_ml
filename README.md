# 超导临界温度 Tc 预测器

基于机器学习的超导临界温度预测 Web 应用。课程大作业。

## 在线演示

🔗 [点击访问](https://你的应用-streamlit.app) <!-- 部署后会填上 -->

## 功能

- 输入化学式（如 `MgB2`、`YBa2Cu3O7`、`LaH10`），预测超导临界温度 Tc
- 自动评估预测置信度（高 / 中 / 低）
- 显示训练集中最相似的化合物作对照
- 可视化预测值在训练数据分布中的位置

## 模型

- **算法**: XGBoost Regressor
- **特征**: Magpie 描述符
- **训练数据**: SuperCon 数据库
- **测试集表现**: R² = 0.918, MAE = 5.48 K

## 已知局限

- 对高压氢化物（如 LaH₁₀、H₃S）的 Tc 预测严重偏低，因训练数据中此类样本极少
- 不考虑压力、晶体结构、电声耦合强度等关键物理参数
- 不区分不同超导机制（BCS / 铜氧化物 / 铁基 / 氢化物）

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 项目结构

```
superc_ml/
├── app.py                  # Streamlit Web UI
├── predict.py              # 预测核心逻辑
├── train_model.py          # 模型训练脚本
├── baseline.py             # 基线模型对比
├── download_data.py        # 数据下载
├── requirements.txt        # 依赖列表
├── data/raw/               # SuperCon 数据
└── models/tc_predictor.pkl # 训练好的模型
```