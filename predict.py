"""
化学式 → Tc 预测的核心逻辑
独立于 UI,可以单独测试
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty


class TcPredictor:
    """超导临界温度预测器"""

    def __init__(self, model_path='models/tc_predictor.pkl',
                 train_data_path='data/raw/unique_m.csv'):
        # 加载模型
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_names = artifact['feature_names']
        self.test_mae = artifact['test_mae']
        self.test_r2 = artifact['test_r2']

        # 加载训练数据(用于查找相似化合物)
        self.train_df = pd.read_csv(train_data_path)[['material', 'critical_temp']]
        self.train_df = self.train_df.drop_duplicates(subset=['material'])

        # 特征化器
        self.featurizer = ElementProperty.from_preset("magpie")

    def predict(self, formula: str) -> dict:
        """
        输入化学式,返回预测结果字典
        包含: Tc 预测值, 置信度, 元素组成, 相似化合物
        """
        result = {
            'formula': formula,
            'success': False,
            'error': None,
            'predicted_tc': None,
            'confidence': None,
            'confidence_reason': None,
            'composition_info': None,
            'similar_compounds': None,
        }

        # 1. 解析化学式
        try:
            comp = Composition(formula)
        except Exception as e:
            result['error'] = f"化学式解析失败: {e}"
            return result

        # 2. 提取特征
        try:
            feats = np.array(self.featurizer.featurize(comp)).reshape(1, -1)
        except Exception as e:
            result['error'] = f"特征提取失败: {e}"
            return result

        # 3. 预测
        pred_tc = float(self.model.predict(feats)[0])
        result['predicted_tc'] = pred_tc

        # 4. 元素组成信息
        result['composition_info'] = {
            str(el): round(amt, 4) for el, amt in comp.get_el_amt_dict().items()
        }

        # 5. 评估置信度
        confidence, reason = self._assess_confidence(comp, pred_tc)
        result['confidence'] = confidence
        result['confidence_reason'] = reason

        # 6. 查找训练集中最相似的化合物
        result['similar_compounds'] = self._find_similar(comp, top_k=5)

        result['success'] = True
        return result

    def _assess_confidence(self, comp, pred_tc):
        """评估预测置信度,返回 (level, reason)"""
        # 检查 1: 预测的 Tc 是否在训练数据熟悉区
        if pred_tc < 0:
            return "low", "预测值为负，物理上不合理"
        if pred_tc > 150:
            return "low", "预测的 Tc 极高,但训练数据中此区域样本极少(模型可能不可靠)"
        if pred_tc > 100:
            return "medium", "预测的 Tc 较高,处于训练数据边界区"

        # 检查 2: 元素是否常见
        elements = set(str(el) for el in comp.elements)
        rare_elements = {'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                         'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf'}
        if elements & rare_elements:
            return "low", f"含有训练数据中罕见的元素: {elements & rare_elements}"

        # 检查 3: 主要是氢的化合物(模型对高压氢化物表现差)
        h_amt = comp.get_atomic_fraction("H") if "H" in [str(e) for e in comp.elements] else 0
        if h_amt > 0.5:
            return "low", f"氢占比 {h_amt:.0%},此类高压氢化物超出训练数据范围"

        return "high", "化学式与训练数据特征相符"

    def _find_similar(self, comp, top_k=5):
        """简单的相似度: 元素集合的 Jaccard 系数 + 元素数量差"""
        target_elements = set(str(el) for el in comp.elements)
        target_n = len(target_elements)

        scores = []
        for _, row in self.train_df.iterrows():
            try:
                c = Composition(row['material'])
                els = set(str(el) for el in c.elements)
                jaccard = len(target_elements & els) / len(target_elements | els)
                # 加一点惩罚:元素数差距越大相似度越低
                n_penalty = 1 / (1 + abs(target_n - len(els)))
                score = jaccard * 0.7 + n_penalty * 0.3
                scores.append((row['material'], row['critical_temp'], score))
            except Exception:
                continue

        scores.sort(key=lambda x: -x[2])
        return [
            {'formula': f, 'tc': float(tc), 'similarity': float(s)}
            for f, tc, s in scores[:top_k]
        ]


# 单独运行用于测试
if __name__ == "__main__":
    print("加载模型...")
    predictor = TcPredictor()
    print(f"模型测试集 R² = {predictor.test_r2:.3f}, MAE = {predictor.test_mae:.2f} K\n")

    test_formulas = ["MgB2", "YBa2Cu3O7", "LaH10", "H3S", "FeSe"]
    for f in test_formulas:
        print(f"\n{'='*50}")
        print(f"预测: {f}")
        print('='*50)
        r = predictor.predict(f)
        if not r['success']:
            print(f"失败: {r['error']}")
            continue
        print(f"预测 Tc: {r['predicted_tc']:.2f} K")
        print(f"置信度: {r['confidence']} ({r['confidence_reason']})")
        print(f"元素组成: {r['composition_info']}")
        print(f"训练集中最相似的化合物:")
        for s in r['similar_compounds']:
            print(f"  {s['formula']:<15} Tc={s['tc']:>6.1f}K, sim={s['similarity']:.2f}")