import openai
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 配置部分 ---
PATH_CONFIG = {
    "MODEL_ENCODER_PATH": "sbert-base-chinese-nli",
    "GOLD_PATH": "../dataset/Complex_Cases/The Beer Murder/Answer.txt",
    "TEST_PATH": "../dataset/Complex_Cases/The Beer Murder/PostRun_KB/report.json"
}

# 相似度阈值
SIMILARITY_THRESHOLD = 0.5

# 加载模型
try:
    MODEL_ENCODER = SentenceTransformer(PATH_CONFIG["MODEL_ENCODER_PATH"])
    print(f"Successfully loaded model")
except Exception as e:
    print(f"FATAL ERROR: Failed to load Sentence Transformer model: {e}")
    MODEL_ENCODER = None

# LLM 客户端配置
LLM_CONFIG = {
    "api_key": "sk-0b72f099f62f463297eb1d4d0d7860a0",
    "base_url": "https://api.deepseek.com/v1",
    "model_name": "deepseek-chat",
    "temperature": 0.0
}
CLIENT = openai.OpenAI(
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"]
)


def invoke_with_messages(messages, is_json=False):
    """通用 LLM 调用函数"""
    response_format_config = {"type": "json_object"} if is_json else {}
    try:
        response = CLIENT.chat.completions.create(
            model=LLM_CONFIG["model_name"],
            messages=messages,
            temperature=LLM_CONFIG["temperature"],
            max_tokens=8000,
            **({"response_format": response_format_config} if is_json else {})
        )

        content = response.choices[0].message.content
        print(content)
        if is_json:
            cleaned_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_content)
        return content
    except Exception as e:
        print(f"LLM调用错误: {e}")
        return {} if is_json else ""

def llm_parse_raw_text_to_json(text):
    """
    利用 LLM 将非结构化文本或列表字符串提取为固定的 JSON 结构
    """
    system_prompt = """
    你是一位专业的案件审查专家。请分析输入的文本（可能是描述性文字或数据列表），提取出最终的定案信息。
    你必须严格按照以下 JSON 格式输出：
    {
        "suspect": "嫌疑人姓名",
        "motive": "详细的作案动机总结",
        "modus_operandi": "详细的作案手法/过程总结"
    }
    """

    user_prompt = f"待分析内容如下：\n\n{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"  >>> 正在调用 LLM 进行文本结构化转换...")
    return invoke_with_messages(messages, is_json=True)

def llm_decompose_text(text):
    """
    将长段的描述文字分解为原子化的事实点
    """
    if not text or not isinstance(text, str):
        return {"atomic_points": []}

    system_prompt = """
    任务：原子信息点分解
    请将提供的长文本分解为一系列**原子化、相互独立、结构完整**的信息点。

    要求：
    1. 每个点必须是完整的陈述句（包含主语、谓语、宾语）。
    2. 主语必须明确（如：'嫌疑人张三' 而不是 '他'）。
    3. 每个句子只表达一个独立的事实。

    严格按照下面的JSON格式输出：
    { "atomic_points": ["事实1", "事实2", ...] }
    """
    user_prompt = f"待分解文本：{text}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return invoke_with_messages(messages, is_json=True)

def calculate_f1(tp, fp, fn):
    """计算 Precision, Recall 和 F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def semantic_f1_matching(model_points, gold_points, threshold, category="Category"):
    """基于语义相似度的贪婪匹配算法"""
    if MODEL_ENCODER is None:
        raise RuntimeError("Sentence Transformer model not loaded.")

    print(f"\n--- {category} 语义匹配 (阈值: {threshold}) ---")

    if not model_points or not gold_points:
        return 0, len(model_points), len(gold_points), []

    model_embeddings = MODEL_ENCODER.encode(model_points)
    gold_embeddings = MODEL_ENCODER.encode(gold_points)

    similarity_matrix = cosine_similarity(model_embeddings, gold_embeddings)

    num_model = len(model_points)
    num_gold = len(gold_points)
    matched_model = np.zeros(num_model, dtype=bool)
    matched_gold = np.zeros(num_gold, dtype=bool)
    tp_count = 0
    match_details = []

    # 贪婪匹配最大相似度对
    while np.sum(~matched_model) > 0 and np.sum(~matched_gold) > 0:
        temp_matrix = similarity_matrix.copy()
        temp_matrix[matched_model, :] = -1
        temp_matrix[:, matched_gold] = -1

        max_similarity = np.max(temp_matrix)
        if max_similarity < threshold:
            break

        max_index = np.unravel_index(np.argmax(temp_matrix), temp_matrix.shape)
        i, j = max_index

        matched_model[i] = True
        matched_gold[j] = True
        tp_count += 1

        print(f"  [Match] Sim={max_similarity:.4f}: '{model_points[i]}' <-> '{gold_points[j]}'")

    fp_count = len(np.where(~matched_model)[0])
    fn_count = len(np.where(~matched_gold)[0])

    return tp_count, fp_count, fn_count, match_details

def solve_optimized(result_json, gold_motive_atoms, gold_mo_atoms):
    """执行完整的评估对比逻辑"""
    print("\n[Step 4] 正在分解待测结果的原子信息点...")
    motive_res = llm_decompose_text(result_json.get("motive", ""))
    mo_res = llm_decompose_text(result_json.get("modus_operandi", ""))

    m_atoms = motive_res.get("atomic_points", [])
    o_atoms = mo_res.get("atomic_points", [])

    # 评估动机
    tp_m, fp_m, fn_m, _ = semantic_f1_matching(m_atoms, gold_motive_atoms, SIMILARITY_THRESHOLD, "动机 (Motive)")
    p_m, r_m, f1_m = calculate_f1(tp_m, fp_m, fn_m)

    # 评估手法
    tp_o, fp_o, fn_o, _ = semantic_f1_matching(o_atoms, gold_mo_atoms, SIMILARITY_THRESHOLD, "作案手法 (MO)")
    p_o, r_o, f1_o = calculate_f1(tp_o, fp_o, fn_o)

    return {
        "motive": {"f1": f1_m, "p": p_m, "r": r_m},
        "mo": {"f1": f1_o, "p": p_o, "r": r_o}
    }


# --- 运行主逻辑 ---
if __name__ == "__main__":
    print("=== 启动自动化案件评估系统 ===")

    # 1. 处理黄金标准 (Answer.txt)
    print(f"\n[Step 1] 加载黄金标准: {PATH_CONFIG['GOLD_PATH']}")
    try:
        with open(PATH_CONFIG["GOLD_PATH"], 'r', encoding='utf-8') as f:
            raw_gold_text = f.read().strip()

        # 结构化黄金标准
        gold_json = llm_parse_raw_text_to_json(raw_gold_text)

        # 原子化黄金标准
        print("  >>> 分解黄金标准事实点...")
        gold_m_atoms = llm_decompose_text(gold_json.get("motive", "")).get("atomic_points", [])
        gold_o_atoms = llm_decompose_text(gold_json.get("modus_operandi", "")).get("atomic_points", [])

    except Exception as e:
        print(f"处理黄金标准失败: {e}")
        exit(1)

    # 2. 读取并转换 report.json (列表转字符串)
    print(f"\n[Step 2] 读取测试文件 (JSON列表): {PATH_CONFIG['TEST_PATH']}")
    try:
        with open(PATH_CONFIG["TEST_PATH"], 'r', encoding='utf-8') as f:
            report_data = json.load(f)

        # 将列表转换为字符串，方便 LLM 理解整体内容
        report_str_for_llm = json.dumps(report_data, ensure_ascii=False)

        # 结构化测试结果
        test_json = llm_parse_raw_text_to_json(report_str_for_llm)

    except Exception as e:
        print(f"处理测试文件失败: {e}")
        exit(1)

    # 3. 执行评估对比
    print(f"\n[Step 3] 开始语义对比评估...")
    try:
        results = solve_optimized(test_json, gold_m_atoms, gold_o_atoms)

        # 4. 打印最终报告
        print("\n" + "=" * 60)
        print("                 案件分析评估报告")
        print("=" * 60)
        print(f"识别嫌疑人: {test_json.get('suspect')}")
        print(f"{'-' * 60}")
        print(f"动机评分 (Motive):")
        print(f"  F1-Score:  {results['motive']['f1']:.4f}")
        print(f"  Precision: {results['motive']['p']:.4f}")
        print(f"  Recall:    {results['motive']['r']:.4f}")
        print(f"{'-' * 60}")
        print(f"手法评分 (Modus Operandi):")
        print(f"  F1-Score:  {results['mo']['f1']:.4f}")
        print(f"  Precision: {results['mo']['p']:.4f}")
        print(f"  Recall:    {results['mo']['r']:.4f}")
        print("=" * 60)

    except Exception as e:
        print(f"评估执行失败: {e}")