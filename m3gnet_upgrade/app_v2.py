"""
app_v2.py — Streamlit Web UI v2 (结构敏感的 Tc 预测器)
=====================================================

3 个 tab:
  1. 结构预测 (M3GNet)  — 上传 cif 或用预设结构, 看到结构感知的预测
  2. 化学式预测 (XGBoost) — 兼容 v1 UI, 只看化学式
  3. 多形态对比          — 同组分不同结构的对比, 是亮点 demo

启动: streamlit run app_v2.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # 找 v1 的 predict.py

# ----- 页面配置 -----
st.set_page_config(
    page_title="Superconductor Tc Predictor v2 (Structure-Aware)",
    page_icon="❄️",
    layout="wide",
)


# =============================================================================
# 1. 缓存模型加载 (Streamlit 会重新运行整个脚本, 缓存避免重复加载)
# =============================================================================

@st.cache_resource
def load_v2_predictor():
    """加载 v2 M3GNet 模型 (大模型, 加载慢, 必须缓存)"""
    try:
        from m3gnet_predictor import M3GNetTcPredictor
        ckpt = ROOT / "models" / "m3gnet_tc.pt"
        cfg = ROOT / "configs" / "train_config.yaml"
        if not ckpt.exists():
            return None, f"未找到训练好的模型: {ckpt}"
        return M3GNetTcPredictor(ckpt_path=ckpt, config_path=cfg), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_v1_predictor():
    """加载 v1 XGBoost 模型"""
    try:
        from predict import TcPredictor
        v1_root = ROOT.parent
        ckpt = v1_root / "models" / "tc_predictor.pkl"
        if not ckpt.exists():
            return None, f"未找到 v1 模型: {ckpt}"
        return TcPredictor(
            model_path=str(ckpt),
            train_data_path=str(v1_root / "data" / "raw" / "unique_m.csv"),
        ), None
    except Exception as e:
        return None, str(e)


# =============================================================================
# 2. 共用工具
# =============================================================================

def render_confidence_badge(conf: str):
    colors = {"high": "#10b981", "medium": "#f59e0b", "low": "#dc2626"}
    labels = {"high": "高置信度 High", "medium": "中等 Medium", "low": "低置信度 Low (外推)"}
    color = colors.get(conf, "#6b7280")
    label = labels.get(conf, conf)
    st.markdown(
        f'<span style="background-color: {color}; color: white; padding: 4px 12px; '
        f'border-radius: 12px; font-size: 14px;">{label}</span>',
        unsafe_allow_html=True,
    )


def render_structure_info(struct: Structure):
    """显示结构的关键信息"""
    spg, _ = struct.get_space_group_info()
    info = {
        "化学式 Formula": str(struct.composition.reduced_formula),
        "原子数 N atoms": len(struct),
        "空间群 Space group": spg,
        "晶格常数 a, b, c (Å)": f"{struct.lattice.a:.3f}, {struct.lattice.b:.3f}, {struct.lattice.c:.3f}",
        "晶格角度 α, β, γ (°)": f"{struct.lattice.alpha:.1f}, {struct.lattice.beta:.1f}, {struct.lattice.gamma:.1f}",
        "体积 Volume (Å³)": f"{struct.volume:.2f}",
    }
    st.dataframe(
        pd.DataFrame({"属性": info.keys(), "值": info.values()}),
        hide_index=True, use_container_width=True,
    )


# =============================================================================
# 3. 预设结构 (3 个多形态对)
# =============================================================================

PRESETS = {
    "Diamond (C, cubic)": lambda: Structure.from_spacegroup(
        "Fd-3m", Lattice.cubic(3.567), ["C"], [[0, 0, 0]]),
    "Graphite (C, hexagonal)": lambda: Structure(
        Lattice.hexagonal(2.461, 6.708),
        ["C"] * 4,
        [[0, 0, 0], [0, 0, 0.5], [1/3, 2/3, 0], [2/3, 1/3, 0.5]],
    ),
    "α-Fe (BCC)": lambda: Structure.from_spacegroup(
        "Im-3m", Lattice.cubic(2.866), ["Fe"], [[0, 0, 0]]),
    "γ-Fe (FCC)": lambda: Structure.from_spacegroup(
        "Fm-3m", Lattice.cubic(3.520), ["Fe"], [[0, 0, 0]]),
    "MgB2 (P6/mmm)": lambda: Structure.from_spacegroup(
        "P6/mmm", Lattice.hexagonal(3.086, 3.524),
        ["Mg", "B"], [[0, 0, 0], [1/3, 2/3, 0.5]]),
    "Nb (BCC)": lambda: Structure.from_spacegroup(
        "Im-3m", Lattice.cubic(3.300), ["Nb"], [[0, 0, 0]]),
}


# =============================================================================
# 4. 页面布局
# =============================================================================

st.title("❄️ Superconductor Tc Predictor")
st.markdown(
    "**v2.0 — Structure-aware**: 用 M3GNet 看到晶体结构, "
    "能区分同组分不同结构 (如石墨 vs 金刚石). 对比 v1 仅看化学式."
)

tab_struct, tab_formula, tab_compare = st.tabs([
    "🔬 结构预测 (v2 M3GNet)",
    "🧪 化学式预测 (v1 XGBoost)",
    "⚖️ 多形态对比",
])


# ============================================================================
# Tab 1: 结构预测
# ============================================================================
with tab_struct:
    st.header("从晶体结构预测 Tc")
    st.markdown(
        "**v2 的新能力**: 模型看到原子位置 + 晶格信息, "
        "因此能区分同组分不同结构的物质."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("输入方式")
        input_mode = st.radio(
            "选择",
            ["上传 cif 文件", "选择预设结构", "粘贴 cif 文本"],
            label_visibility="collapsed",
        )

        structure = None

        if input_mode == "上传 cif 文件":
            uploaded = st.file_uploader("拖入或选择 cif", type=["cif"])
            if uploaded:
                cif_str = uploaded.read().decode("utf-8")
                try:
                    structure = Structure.from_str(cif_str, fmt="cif")
                    st.success(f"✓ 成功解析 {uploaded.name}")
                except Exception as e:
                    st.error(f"解析失败: {e}")

        elif input_mode == "选择预设结构":
            choice = st.selectbox("预设结构 (含著名多形态)", list(PRESETS.keys()))
            if choice:
                structure = PRESETS[choice]()

        elif input_mode == "粘贴 cif 文本":
            cif_text = st.text_area("粘贴 cif 格式的文本", height=200)
            if cif_text.strip():
                try:
                    structure = Structure.from_str(cif_text, fmt="cif")
                    st.success("✓ 解析成功")
                except Exception as e:
                    st.error(f"解析失败: {e}")

    with col2:
        if structure is not None:
            st.subheader("结构信息")
            render_structure_info(structure)

            if st.button("🚀 预测 Tc", type="primary"):
                v2_pred, err = load_v2_predictor()
                if err:
                    st.error(f"模型加载失败: {err}")
                else:
                    with st.spinner("预测中..."):
                        try:
                            result = v2_pred.predict_structure(structure)
                        except Exception as e:
                            st.error(f"预测失败: {e}")
                            result = None

                    if result:
                        st.markdown("### 预测结果")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Predicted Tc (K)", f"{result['tc_pred']:.2f}")
                        with c2:
                            st.markdown("**置信度**")
                            render_confidence_badge(result["confidence"])

                        if result["confidence"] == "low":
                            st.warning(
                                "⚠ 此预测置信度低. 可能原因: "
                                "Tc 超出训练分布; 含有罕见元素; 氢比例过高."
                            )


# ============================================================================
# Tab 2: 化学式预测 (v1)
# ============================================================================
with tab_formula:
    st.header("从化学式预测 Tc (v1 兼容模式)")
    st.markdown(
        "v1 的 XGBoost + Magpie 路径. **注意: 这一模式无法区分同组分不同结构.** "
        "如需结构敏感预测, 用 v2 tab."
    )

    formula = st.text_input("化学式", value="MgB2",
                            help="例如: MgB2, YBa2Cu3O7, LaH10")

    if st.button("🧪 v1 预测", key="v1_predict"):
        v1_pred, err = load_v1_predictor()
        if err:
            st.error(f"v1 模型加载失败: {err}")
        elif formula.strip():
            try:
                result = v1_pred.predict(formula.strip())
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Predicted Tc (K)", f"{result['tc_pred']:.2f}")
                with c2:
                    st.markdown("**置信度**")
                    render_confidence_badge(result.get("confidence", "medium"))
            except Exception as e:
                st.error(f"预测失败: {e}")


# ============================================================================
# Tab 3: 多形态对比 (亮点功能)
# ============================================================================
with tab_compare:
    st.header("多形态对比: 看 v1 vs v2 的差异")
    st.markdown(
        "**这是 v2 升级最有说服力的演示.** 同组分不同结构: "
        "v1 给出相同预测, v2 给出不同预测."
    )

    pair_choice = st.selectbox(
        "选择对比对",
        ["Diamond vs Graphite (C)", "α-Fe vs γ-Fe (Fe)"],
    )

    if pair_choice == "Diamond vs Graphite (C)":
        s1, s2 = PRESETS["Diamond (C, cubic)"](), PRESETS["Graphite (C, hexagonal)"]()
        n1, n2 = "Diamond", "Graphite"
    else:
        s1, s2 = PRESETS["α-Fe (BCC)"](), PRESETS["γ-Fe (FCC)"]()
        n1, n2 = "α-Fe", "γ-Fe"

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(n1)
        render_structure_info(s1)
    with col2:
        st.subheader(n2)
        render_structure_info(s2)

    if st.button("🚀 对比预测", type="primary", key="compare"):
        v1_pred, _ = load_v1_predictor()
        v2_pred, _ = load_v2_predictor()

        rows = []
        for name, s in [(n1, s1), (n2, s2)]:
            row = {"Structure": name, "Formula": str(s.composition.reduced_formula)}
            if v1_pred:
                try:
                    row["v1 Tc (XGBoost)"] = f"{v1_pred.predict(row['Formula'])['tc_pred']:.2f} K"
                except Exception:
                    row["v1 Tc (XGBoost)"] = "N/A"
            if v2_pred:
                try:
                    row["v2 Tc (M3GNet)"] = f"{v2_pred.predict_structure(s)['tc_pred']:.2f} K"
                except Exception:
                    row["v2 Tc (M3GNet)"] = "N/A"
            rows.append(row)

        st.markdown("### 结果对比")
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.info(
            "**观察**: v1 (XGBoost) 对两个多形态给出**相同**预测, 因为它只看化学式. "
            "v2 (M3GNet) 给出**不同**预测, 因为它看到了晶体结构. "
            "这就是老师提出 '石墨/金刚石化学式相同, Tc 是否一样' 问题的答案."
        )


# ----- 侧边栏 -----
with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "v2.0 升级: 用 M3GNet (graph neural net) 替换 Magpie 组分描述符. "
        "训练数据集从 SuperCon (21k 条) 切换到 3DSC (~5.7k 条, 带结构)."
    )
    st.markdown("---")
    st.markdown("**v1 baseline**: XGBoost + Magpie 132 features")
    st.markdown("**v2 model**: M3GNet (3-body GNN, fine-tuned)")
    st.markdown("**Dataset v2**: 3DSC (Sommer et al., Nature SciData 2023)")
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/Kruger114514/superc_ml)")
