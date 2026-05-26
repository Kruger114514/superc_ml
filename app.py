"""
超导临界温度 Tc 预测 - Web UI (含 Agent 模块)
运行: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from predict import TcPredictor


# ===== 页面配置 =====
st.set_page_config(
    page_title="超导 Tc 预测",
    page_icon="🧲",
    layout="wide"
)


# ===== 加载模型 (缓存) =====
@st.cache_resource
def load_predictor():
    return TcPredictor()


@st.cache_data
def load_train_data():
    return pd.read_csv('data/raw/unique_m.csv')[['material', 'critical_temp']].drop_duplicates(subset=['material'])


@st.cache_resource
def load_agent():
    """Agent 单独缓存,因为加载比较慢"""
    from agent import SuperconductorAgent
    return SuperconductorAgent()


predictor = load_predictor()
train_df = load_train_data()


# ===== 页面标题 =====
st.title("🧲 超导临界温度 Tc 预测")
st.markdown(
    f"基于 XGBoost + Magpie 特征,在 SuperCon 数据集上训练。"
    f"**测试集 R² = {predictor.test_r2:.3f}, MAE = {predictor.test_mae:.2f} K**"
)


# ===== Tab 导航 =====
tab_predict, tab_agent, tab_about = st.tabs(["🔮 单体预测", "🤖 智能筛选 (Agent)", "ℹ️ 关于"])


# ============== Tab 1: 单体预测 ==============
with tab_predict:
    col_input, col_examples = st.columns([2, 1])

    with col_input:
        st.subheader("输入化学式")
        formula = st.text_input(
            "",
            value="MgB2",
            placeholder="例如: MgB2, YBa2Cu3O7, LaH10",
            label_visibility="collapsed",
            key="predict_formula"
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

    if predict_btn and formula:
        result = predictor.predict(formula)

        if not result['success']:
            st.error(f"❌ {result['error']}")
        else:
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


# ============== Tab 2: 智能筛选 Agent ==============
with tab_agent:
    st.subheader("🤖 自然语言智能筛选")
    st.markdown(
        "输入自然语言需求, Agent 会自动解析、从训练集中筛选候选化合物、"
        "调用预测模型、生成分析报告。"
    )

    # 初始化 session state
    if 'agent_query_text' not in st.session_state:
        st.session_state['agent_query_text'] = '找几个含铜和氧的高温超导体'

    # 示例按钮
    st.markdown("**快速试用:**")
    cols = st.columns(3)
    example_queries = [
        "找几个含铜和氧的高温超导体",
        "推荐二元的铁基超导体",
        "不含氧的化合物中预测 Tc 较高的",
    ]
    for i, eq in enumerate(example_queries):
        with cols[i]:
            if st.button(eq, use_container_width=True, key=f"eq_{i}"):
                st.session_state['agent_query_text'] = eq
                st.rerun()

    # text_area 直接绑定 session state
    query = st.text_area(
        "你的需求:",
        height=80,
        key='agent_query_text',
    )

    if st.button("🚀 启动 Agent", type="primary"):
        with st.spinner("Agent 工作中... (大约 30 秒)"):
            try:
                agent = load_agent()
                result = agent.run(query)
            except Exception as e:
                st.error(f"Agent 初始化失败: {e}\n\n请确认 .env 中的 DEEPSEEK_API_KEY 已设置。")
                result = None

        if result is not None:
            if result['error']:
                st.error(f"运行出错: {result['error']}")
            else:
                # 显示解析的条件
                with st.expander("📋 Agent 解析的筛选条件 (点击展开)"):
                    st.json(result['criteria'])

                # 显示 Agent 分析
                if result['summary']:
                    st.success(f"**🤖 Agent 分析:**\n\n{result['summary']}")

                # 显示候选列表
                if result['top_results'] is not None and not result['top_results'].empty:
                    st.subheader("📊 Top 候选化合物")
                    display_df = result['top_results'].copy()
                    display_df.columns = ['化学式', '实验 Tc (K)', '预测 Tc (K)', '置信度']
                    display_df['实验 Tc (K)'] = display_df['实验 Tc (K)'].round(1)
                    display_df['预测 Tc (K)'] = display_df['预测 Tc (K)'].round(1)
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("未找到符合条件的化合物,试试放宽筛选条件。")


# ============== Tab 3: 关于 ==============
with tab_about:
    st.markdown(f"""
    ## 关于本项目

    **小组成员** ：宋奡  王祉懿 高鹏 陈梦婷 周祎玮           
    **模型**: XGBoost Regressor
    **特征**: Magpie (132 个组分描述符)
    **训练数据**: SuperCon (Hamidieh 2018)
    **测试集表现**:
    - R² = {predictor.test_r2:.3f}
    - MAE = {predictor.test_mae:.2f} K

    **Agent 模块**: 基于 DeepSeek LLM,实现自然语言到结构化查询的转换,
    并自动调用预测模型完成筛选。

    ### 已知局限
    - 对 Tc > 150K 的高压氢化物预测严重偏低
    - 不考虑压力、晶体结构等物理量
    - 不区分常规 BCS、铜氧化物、铁基等不同机制

    ### GitHub
    https://github.com/Kruger114514/superc_ml
    """)