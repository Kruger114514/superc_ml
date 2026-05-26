"""
超导体筛选 Agent
功能: 接收自然语言请求,自动筛选候选化合物并预测 Tc
"""
import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from pymatgen.core import Composition
from predict import TcPredictor

load_dotenv()


class SuperconductorAgent:
    """超导体筛选 Agent"""

    def __init__(self):
    # 加载 LLM - 优先用 Streamlit secrets, 否则用 .env
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("DEEPSEEK_API_KEY", None)
    except Exception:
        pass  # 不在 streamlit 环境里就跳过

    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置, 检查 .env 或 Streamlit Secrets")
        self.llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # 加载 Tc 预测器
        self.predictor = TcPredictor()

        # 加载训练集 (用于初步候选筛选)
        self.train_df = pd.read_csv('data/raw/unique_m.csv')[['material', 'critical_temp']]
        self.train_df = self.train_df.drop_duplicates(subset=['material']).reset_index(drop=True)

        # 解析每个化学式的元素集合 (一次性预处理)
        def get_elements(formula):
            try:
                return set(str(e) for e in Composition(formula).elements)
            except Exception:
                return set()

        self.train_df['elements'] = self.train_df['material'].apply(get_elements)
        self.train_df['n_elements'] = self.train_df['elements'].apply(len)

    def parse_query(self, user_query: str) -> dict:
        """step 1: 用 LLM 把自然语言解析为结构化查询"""
        system_prompt = """你是材料科学查询解析器。把用户的自然语言请求转换为 JSON 格式的筛选条件。

输出格式必须是合法 JSON, 包含以下字段:
- required_elements: 数组,必须包含的元素符号列表 (如 ["Cu", "O"]),可以为空 []
- forbidden_elements: 数组,不能包含的元素符号列表,可以为空 []
- min_tc: 数字,最低 Tc 临界值 (K),无要求时为 0
- max_tc: 数字,最高 Tc 上限 (K),无要求时为 1000
- n_elements_range: 数组 [min, max],化合物的元素数量范围,默认 [2, 6]
- description: 字符串,用一句中文复述需求

只输出 JSON,不要任何其他文字、Markdown 标记、解释。"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def filter_candidates(self, criteria: dict, top_k: int = 50) -> pd.DataFrame:
        """step 2: 从训练集筛选符合条件的候选"""
        df = self.train_df.copy()

        required = set(criteria.get('required_elements', []))
        forbidden = set(criteria.get('forbidden_elements', []))
        n_min, n_max = criteria.get('n_elements_range', [2, 6])

        if required:
            df = df[df['elements'].apply(lambda s: required.issubset(s))]
        if forbidden:
            df = df[df['elements'].apply(lambda s: not (forbidden & s))]
        df = df[df['n_elements'].between(n_min, n_max)]

        return df.head(top_k * 2)  # 多取一些,后面预测后再排序

    def predict_batch(self, candidates: pd.DataFrame, criteria: dict) -> pd.DataFrame:
        """step 3: 批量预测 Tc, 按预测值排序"""
        results = []
        for _, row in candidates.iterrows():
            r = self.predictor.predict(row['material'])
            if r['success']:
                results.append({
                    'formula': row['material'],
                    'experimental_tc': row['critical_temp'],
                    'predicted_tc': r['predicted_tc'],
                    'confidence': r['confidence'],
                })

        df = pd.DataFrame(results)
        if df.empty:
            return df

        # 按 min_tc 过滤
        min_tc = criteria.get('min_tc', 0)
        max_tc = criteria.get('max_tc', 1000)
        df = df[(df['predicted_tc'] >= min_tc) & (df['predicted_tc'] <= max_tc)]

        return df.sort_values('predicted_tc', ascending=False).head(10)

    def summarize(self, user_query: str, criteria: dict, top_results: pd.DataFrame) -> str:
        """step 4: 让 LLM 生成自然语言总结"""
        if top_results.empty:
            return "未找到符合条件的化合物。可以放宽筛选条件再试。"

        top_str = top_results.to_string(index=False)
        system_prompt = """你是材料科学分析助手。基于用户需求和筛选结果,用中文写一段简短分析 (3-5 句)。

要点:
- 提到找到了多少候选,Top 候选是什么
- 提一下预测 Tc 范围
- 如果有低置信度的结果, 提醒一下
- 不要重复表格内容, 用文字总结
- 简洁专业"""

        user_content = f"""用户需求: {user_query}
解析的筛选条件: {json.dumps(criteria, ensure_ascii=False)}
Top 候选 (按预测 Tc 降序):
{top_str}

请写分析:"""

        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content

    def run(self, user_query: str) -> dict:
        """主流程: 端到端处理一次请求"""
        result = {
            'query': user_query,
            'criteria': None,
            'candidates': None,
            'top_results': None,
            'summary': None,
            'error': None,
        }

        try:
            criteria = self.parse_query(user_query)
            result['criteria'] = criteria

            candidates = self.filter_candidates(criteria)
            result['candidates'] = candidates
            if candidates.empty:
                result['summary'] = "训练集中没有符合元素筛选条件的化合物。"
                return result

            top_results = self.predict_batch(candidates, criteria)
            result['top_results'] = top_results

            summary = self.summarize(user_query, criteria, top_results)
            result['summary'] = summary
        except Exception as e:
            result['error'] = str(e)

        return result


# 命令行测试
if __name__ == "__main__":
    print("初始化 Agent...")
    agent = SuperconductorAgent()
    print("初始化完成\n")

    test_queries = [
        "帮我找一个含 Cu 和 O 的高温超导体, Tc 大于 80K",
        "推荐几个二元的铁基超导体",
        "找一些不含氧的超导体, 预测 Tc 比较高的",
    ]

    for q in test_queries:
        print("=" * 60)
        print(f"用户: {q}")
        print("=" * 60)
        result = agent.run(q)
        if result['error']:
            print(f"错误: {result['error']}")
            continue
        print(f"\n[解析的条件]")
        print(json.dumps(result['criteria'], ensure_ascii=False, indent=2))
        print(f"\n[Top 候选]")
        if result['top_results'] is not None and not result['top_results'].empty:
            print(result['top_results'].to_string(index=False))
        else:
            print("(无)")
        print(f"\n[Agent 分析]")
        print(result['summary'])
        print()