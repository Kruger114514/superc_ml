"""
超导临界温度 Tc 预测器 - Web UI
运行: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predict import TcPredictor


# ===== 页面配置 =====
st.set_page_config(
    page_title="超导 Tc 预测器",
    page_icon="🧲",
    layout="wide"
)


# ===== 加载模型 (缓存,只加载一次) =====
@st.cache_resource
def load_predictor():
    return TcPredictor()


@st.cache_data
def load_train_data():
    return pd.read_csv('data/raw/unique_m.csv')[['material', 'critical_temp']].drop_duplicates(subset=['material'])


predictor = load_predictor()
train_df = load_train_data()


# ===== 页面标题 =====
st.title("🧲 超导临界温度 Tc 预测器")
st.markdown(
    f"基于 XGBoost + Magpie 特征,在 SuperCon 数据集上训练。"
    f"**测试集 R² = {predictor.test_r2:.3f}, MAE = {predictor.test_mae:.2f} K**"
)
st.divider()


# ===== 输入区 =====
col_input, col_examples = st.columns([2, 1])

with col_input:
    st.subheader("输入化学式")
    formula = st.text_input(
        "",
        value="MgB2",
        placeholder="例如: MgB2, YBa2Cu3O7, LaH10",
        label_visibility="collapsed"
    )
    predict_btn = st.button("🔮 预测 Tc", type="primary", use_container_width=True)

with col_examples:
    st.subheader("快速试用")
    examples = {
        "MgB2 (BCS, 39K)": "MgB2",
        "YBa₂Cu₃O₇ (高温, 92K)": "YBa2Cu3O7",
        "FeSe (铁基, 8K)": "FeSe",
        "LaH₁₀ (氢化物, 250K)": "LaH10",
        "H₃S (氢化物, 203K)": "H3S",
    }
    for label, f in examples.items():
        if st.button(label, use_container_width=True, key=f"ex_{f}"):
            formula = f
            predict_btn = True


# ===== 预测和展示 =====
if predict_btn and formula:
    result = predictor.predict(formula)

    if not result['success']:
        st.error(f"❌ {result['error']}")
    else:
        # === 顶部大数字 + 置信度 ===
        col1, col2, col3 = st.columns([1.5, 1, 1.5])

        with col1:
            st.metric(
                label="预测临界温度",
                value=f"{result['predicted_tc']:.1f} K",
                delta=f"{result['predicted_tc'] - 273.15:.1f} °C",
                delta_color="off"
            )

        with col2:
            conf = result['confidence']
            conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}[conf]
            conf_text = {"high": "高置信度", "medium": "中等置信度", "low": "低置信度"}[conf]
            st.metric("置信度", f"{conf_emoji} {conf_text}")

        with col3:
            st.markdown("**置信度说明:**")
            st.info(result['confidence_reason'])

        if conf == "low":
            st.warning(
                "⚠️ **重要提示**: 此预测结果不可靠。模型训练数据中此类化合物极少,"
                "实际 Tc 可能与预测值相差极大。建议参考下方的相似化合物作对照。"
            )

        st.divider()

        # === 元素组成 + 相似化合物 ===
        col_comp, col_sim = st.columns(2)

        with col_comp:
            st.subheader("📐 元素组成")
            comp_df = pd.DataFrame([
                {"元素": el, "原子数": amt}
                for el, amt in result['composition_info'].items()
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        with col_sim:
            st.subheader("🔍 训练集中最相似的化合物")
            sim_df = pd.DataFrame([
                {"化学式": s['formula'],
                 "实验 Tc (K)": round(s['tc'], 1),
                 "相似度": round(s['similarity'], 2)}
                for s in result['similar_compounds']
            ])
            st.dataframe(sim_df, hide_index=True, use_container_width=True)

        st.divider()

        # === 在训练数据分布上标记预测位置 ===
        st.subheader("📊 预测位置在训练数据中的相对分布")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(train_df['critical_temp'], bins=80, alpha=0.6,
                color='steelblue', label='Training data')
        ax.axvline(result['predicted_tc'], color='red', linewidth=2.5,
                   linestyle='--', label=f'Predicted: {result["predicted_tc"]:.1f} K')
        ax.set_xlabel('Tc (K)')
        ax.set_ylabel('Count')
        ax.set_title(f'{formula} prediction position in training distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(
            "⬆️ 灰色直方图是 21000+ 训练样本的 Tc 分布,红色虚线是你输入化学式的预测值。"
            "如果红线落在直方图稀疏的右侧,说明模型在外推,结果不可靠。"
        )


# ===== 侧边栏:模型信息 =====
with st.sidebar:
    st.header("ℹ️ 关于本项目")
    st.markdown(f"""
**模型**: XGBoost Regressor
**特征**: Magpie 
**训练数据**: SuperCon 
**测试集表现**:
- R² = {predictor.test_r2:.3f}
- MAE = {predictor.test_mae:.2f} K

**已知局限**:
- 对 Tc > 150K 的高压氢化物预测严重偏低
- 不考虑压力、晶体结构等物理量
- 不区分常规 BCS、铜氧化物、铁基等不同机制

**作者**: 宋奡 王祉懿 高鹏 陈梦婷
**课程**: 数据驱动大作业
    """)