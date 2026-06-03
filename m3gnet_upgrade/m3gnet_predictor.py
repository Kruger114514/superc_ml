"""
m3gnet_predictor.py — 封装好的 M3GNet Tc 预测器
================================================

把训练好的 M3GNet 包装成一个类, 提供两个接口:
  - predict_structure(structure)         <- 从 pymatgen.Structure 预测
  - predict_from_cif(cif_path_or_str)    <- 从 cif 文件或字符串预测

被 app_v2.py (Streamlit) 和 scripts/05_predict_polymorphs.py 调用。

复用 v1 风格: 类 + dict 返回结果 + confidence 评估。
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import yaml
from pymatgen.core import Structure

warnings.simplefilter("ignore")


class M3GNetTcPredictor:
    """
    用 fine-tuned M3GNet 预测晶体结构的 Tc.

    Usage:
        predictor = M3GNetTcPredictor("models/m3gnet_tc.pt", "configs/train_config.yaml")
        result = predictor.predict_structure(some_pymatgen_structure)
        print(result["tc_pred"], result["confidence"])
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        config_path: Union[str, Path],
        device: str = "auto",
    ):
        self.ckpt_path = Path(ckpt_path)
        self.config_path = Path(config_path)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        with open(self.config_path) as f:
            self.cfg = yaml.safe_load(f)

        self._load_model()
        self._setup_converter()

    def _load_model(self):
        """加载训练好的 Lightning checkpoint"""
        # 动态加载 03_train_m3gnet_tc.py 复用 build_model + TcRegressor
        train_script = Path(__file__).parent / "scripts" / "03_train_m3gnet_tc.py"
        spec = importlib.util.spec_from_file_location("train_module", train_script)
        train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mod)

        model = train_mod.build_model(self.cfg)
        self.lit_module = train_mod.TcRegressor.load_from_checkpoint(
            str(self.ckpt_path), model=model, map_location=self.device,
        )
        self.lit_module.eval()
        self.lit_module.to(self.device)

    def _setup_converter(self):
        """初始化 Structure -> graph 转换器"""
        from matgl.config import DEFAULT_ELEMENTS
        from matgl.ext.pymatgen import Structure2Graph

        self.converter = Structure2Graph(
            element_types=DEFAULT_ELEMENTS,
            cutoff=self.cfg["model"]["cutoff"],
        )
        self.threebody_cutoff = self.cfg["model"]["threebody_cutoff"]
        self.log_transform = self.cfg["data"]["log_transform"]

    def predict_structure(self, structure: Structure) -> Dict:
        """
        从 pymatgen.Structure 预测 Tc.
        返回 dict: {tc_pred, confidence, n_atoms, formula, ...}
        """
        # 转 graph
        graph, lattice, state_attr = self.converter.get_graph(structure)

        # 构造 line graph (M3GNet 三体相互作用需要)
        try:
            import dgl
            line_graph = dgl.line_graph(graph, shared=True)
            # 计算三体 cutoff 内的边
            # 实际上 matgl 的 collate_fn_graph 处理这个, 这里简化
        except ImportError:
            line_graph = None

        graph = graph.to(self.device)
        if line_graph is not None:
            line_graph = line_graph.to(self.device)

        with torch.no_grad():
            pred = self.lit_module(graph, line_graph, state_attr)
            pred = pred.squeeze().cpu().numpy()

        # 反 log
        if self.log_transform:
            tc_pred = float(np.exp(pred) - 1)
        else:
            tc_pred = float(pred)
        tc_pred = max(0, tc_pred)  # Tc 不能负

        # 置信度评估 (复用 v1 思路)
        confidence = self._assess_confidence(structure, tc_pred)

        return {
            "tc_pred": tc_pred,
            "confidence": confidence,
            "formula": str(structure.composition.reduced_formula),
            "n_atoms": len(structure),
            "volume": float(structure.volume),
            "spacegroup": structure.get_space_group_info()[0],
            "lattice_params": {
                "a": float(structure.lattice.a),
                "b": float(structure.lattice.b),
                "c": float(structure.lattice.c),
                "alpha": float(structure.lattice.alpha),
                "beta": float(structure.lattice.beta),
                "gamma": float(structure.lattice.gamma),
            },
        }

    def predict_from_cif(self, cif_input: Union[str, Path]) -> Dict:
        """
        从 cif 文件路径或 cif 字符串内容预测.
        """
        cif_input = str(cif_input)
        if cif_input.endswith(".cif") and Path(cif_input).exists():
            structure = Structure.from_file(cif_input)
        else:
            # 当成 cif 字符串内容处理
            structure = Structure.from_str(cif_input, fmt="cif")
        return self.predict_structure(structure)

    def _assess_confidence(self, structure: Structure, tc_pred: float) -> str:
        """
        基于多个因素评估预测的置信度.
        """
        # 因素 1: Tc 范围 (训练集主要在 0-140K)
        if tc_pred > 150:
            return "low"
        if tc_pred > 100:
            return "medium"

        # 因素 2: 元素罕见度
        elements = [str(s) for s in structure.composition.elements]
        rare_elements = {"Tc", "Pm", "Pa", "Po", "At", "Fr", "Ra", "Ac"}
        if any(e in rare_elements for e in elements):
            return "low"

        # 因素 3: 含 H 比例 (氢化物外推)
        comp = structure.composition.fractional_composition.as_dict()
        h_frac = comp.get("H", 0) + comp.get("H+", 0)
        if h_frac > 0.5:
            return "low"

        return "high"


if __name__ == "__main__":
    # 简单测试
    print("Testing M3GNetTcPredictor...")
    root = Path(__file__).parent
    predictor = M3GNetTcPredictor(
        ckpt_path=root / "models" / "m3gnet_tc.pt",
        config_path=root / "configs" / "train_config.yaml",
    )

    # 测试: MgB2
    from pymatgen.core import Lattice
    mgb2 = Structure.from_spacegroup(
        "P6/mmm", Lattice.hexagonal(3.086, 3.524),
        ["Mg", "B"], [[0, 0, 0], [1/3, 2/3, 0.5]],
    )
    result = predictor.predict_structure(mgb2)
    print(f"MgB2: predicted Tc = {result['tc_pred']:.2f} K "
          f"({result['confidence']} confidence)")
