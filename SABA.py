import openai
import json
import os


T_MAX=2
PATH_CONFIG={
    "INPUT_PATH": "CASE/Mystery_text.txt",
    "OUTPUT_QA_PATH": "CASE/SABA_PostRun_KB/q_a.json",
    "OUTPUT_REPORT_PATH": "CASE/SABA_PostRun_KB/report.json"
}
TOTAL_USAGE = {"input": 0, "output": 0, "hit": 0}


LLM_CONFIG = {
    "api_key": "YOUR_API_KEY",
    "base_url": "BASE_URL",
    "model_name": "MODE_NAME",
    "temperature": 0.0
}

CLIENT = openai.OpenAI(
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"]
)
def invoke_with_messages(messages, is_json=False):
    print("--------当前输入为-----------")
    print(messages)
    response_format_config = {"type": "json_object"} if is_json else {}
    response = CLIENT.chat.completions.create(
        model=LLM_CONFIG["model_name"],
        messages=messages,
        temperature=LLM_CONFIG["temperature"],
        max_tokens=8000,
        **({"response_format": response_format_config} if is_json else {})
    )

    # 提取 Token 使用情况
    TOTAL_USAGE["input"] += response.usage.prompt_tokens
    TOTAL_USAGE["output"] += response.usage.completion_tokens
    TOTAL_USAGE["hit"] += response.usage.prompt_cache_hit_tokens

    content = response.choices[0].message.content
    print("--------当前输出为-----------")
    print(content)
    print(f"{'=' * 60}")
    print(f"输入token{response.usage.prompt_tokens}")
    print(f"输出token{response.usage.completion_tokens}")
    print(f"命中token{response.usage.prompt_cache_hit_tokens}")
    print(f"{'=' * 60}")

    if is_json:
        cleaned_content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_content)
    return content

def saba_if(info):
    system_prompt = f"""
    [角色]:资深文本分析专家,具备严谨的逻辑思维和全面的信息处理能力
    [核心任务]:
      - 第一阶段：将案件文本拆解为逻辑主轴(核心叙事+时间线)和碎片化属性(嫌疑人/证人/场景/物证/将文本中的各类信息分解为独立的、可检索的属性单元);
        - [输出格式如下]: 
          - 1.逻辑主轴(包含叙事的主线脉络及关键转折点、主要事件的时间序列、事件之间的因果关系):
            - 核心事件链条: ["事件1描述", "事件2描述", ...],
            - 关键时间节点: ["时间点1: 相关事件", "时间点2: 相关事件", ...],
            - 主要因果关系: ["因 → 果描述", ...]
          - 2.碎片化属性分解(文本分解为碎片化属性,确保全面性与完整性,不得遗漏任何相关信息):
            - 人物属性: ["人物1名字对话与互动、执行的具体动作或行动以及所有的相关信息,按时间线索整理和归纳",]
            - 场景属性: ["追踪每个场景和物品在不同时间点的具体状态,静态结构、宏观环境、微观状态分开记录",]
            - 其他属性: ...
      - 第二阶段：对第一阶段提取的信息进行整合,并完成语义标注（疑点/矛盾点批注）.
        - [输出格式如下]: 
          - 1.特征对齐(以事件时间线为逻辑主轴,将碎片化属性融入对应时间节点):
            - 物品信息补充：将物品的详细描述、发现位置补充到对应事件中
            - 人物信息关联：将人物性格与行为对比、证词与行为对比（标注印证/矛盾）.
          - 2.语义标注(疑点批注):
            - 对整合后的每条时间线、物证信息添加批注（格式：【批注：具体疑点/矛盾点/需深挖的线索】）,批注需基于常识和逻辑推理.
    [必须严格遵守的规则]:
      - 必须完整,无遗漏原始文本中的关键内容,不做主观筛选;
      - 事件描述严格保留原文,不增删、不修改;
    """
    user_prompt = f"""
    [案件信息]:{info}
    请开始你的分析
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    content = invoke_with_messages(messages)
    return content

def saba_h(info,obstacles):
    def get_messages(q_a,obstacle):
        system_prompt = f"""
        [角色]:你是资深侦探,需处理认知障碍(阻碍案件推理的关键问题)
        [要求按以下要求处理每个障碍]:
          - 步骤1:子问题分解
            - 每个障碍分解为1-3个具体、可调查的子问题,子问题需符合"动机/手法/时间/空间"维度,避免笼统;
          - 步骤2:假设生成
            - 基于案件信息和证据链,为每个子问题生成符合常识的侦探假设(需满足:1. 时间线一致；2. 空间位置合理；3. 符合物理定律)假设需具体,避免模糊表述.
        [严格按照下面的JSON格式输出]: 
        {{
          "q_a":[
            {{
              "q":"子问题1",
              "a":"子问题1假设"
            }},
            //根据实际添加子问题
            {{
              "q":最后一个必须是[当前障碍],
              "a":"问题假设"
            }}
          ]
        }}
        """
        user_prompt = f"""
        [案件信息]:{info}
        [当前障碍]:{obstacle}
        [已有证据链]：{"\n".join([f"问题：{item['q']} → 假设：{item['a']}" for item in q_a]) if q_a else "无"}
        请开始你的分析
        """
        return  [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    q_a=[]
    for obstacle in obstacles:
        content = invoke_with_messages(get_messages(q_a,obstacle),is_json=True)
        q_a.extend(content.get("q_a",[]))

    file_path=PATH_CONFIG["OUTPUT_QA_PATH"]
    history = json.load(open(file_path, 'r', encoding='utf-8')) if os.path.exists(file_path) and os.path.getsize(
        file_path) > 0 else []
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history + q_a, f, ensure_ascii=False, indent=4)
    return q_a


def saba_q(info,current_task):
    def get_messages(d_t,point):
        system_prompt = f"""
        [角色]:你是一个喜欢<一步一步思考>的专业推理分析师,当前任务是：{current_task}
        [要求]:
          - 评估当前信息是否足够支撑完美的完成任务,若不足,识别所有阻碍推理的认知障碍;
          - 障碍需满足：具体、可拆解、与案件核心逻辑相关(如"动机不明确"→"嫌疑人A的杀人直接导火索未验证/嫌疑人A的作案工具获取途径未验证/嫌疑人B的时间线存在矛盾");
        [严格按照下面的JSON格式输出]:
        - 当仍然存在认知障碍时:
        {{
            "obstacles": [
              "障碍1",
              "障碍2",
              //根据实际添加(严格遵守: 最多8个障碍)
            ]
        }}
        - 当你认为根据已有信息足以完美的完成任务时:
        {{
            "report":""
        }}
        """
        user_prompt = f"""
    [案件信息]:{info}
    [已有证据链]：{"\n".join([f"问题：{item['q']} → 假设：{item['a']}" for item in d_t]) if d_t else "无"}
    请开始你的分析
    """
        if point == 1: user_prompt += "**重点:此次输出必须认为根据已有信息足以完美的完成任务并输出report,不可输出障碍列表**"
        return  [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    d_t=[]
    t=0
    while t<T_MAX:
        t += 1
        content = invoke_with_messages(get_messages(d_t, t==T_MAX), is_json=True)
        if "report" in content and content.get("report"):
            return content.get("report", "")
        elif "obstacles" in content:
            result = saba_h(info, content.get("obstacles", []))
            d_t.extend(result)
        else:
            print(f"警告：第 {t} 次迭代返回格式不符合预期")
    return d_t

def saba_qsr(info):
    all_task=[
        "对每个嫌疑人进行动机剖析,纯粹的分析与呈现完成一份‘动机报告’,一个人一个人地列清楚.详细阐述每个嫌疑人动机的逻辑链条、强烈程度以及其中的合理性与薄弱点.不比较,不排除,不总结.",
        "对每个嫌疑人进行作案手法与机会剖析,纯粹的分析与呈现完成一份‘作案手法与机会报告’,一个人一个人地列清楚.详细分析每个嫌疑人‘谁可能做到’,即他有没有办法拿到/用到作案工具？或者他有没有机会单独作案？不比较,不排除,不总结.",
        "确定唯一凶手,纯粹的分析与呈现完成一份‘最终报告’,确保原始信息中所有反常信息你都有一个合理的解释,比如“是作者故意留下的干扰信息/此信息才是真正确定真凶的关键/此信息完美排除了..”"
    ]
    history_report=[]
    for current_task in all_task:
        history_str = "\n".join([str(i) for i in history_report]) if history_report else "无"
        result = saba_q(f"{info}\n已有信息：{history_str}", current_task)
        history_report.append(result)

    with open(PATH_CONFIG["OUTPUT_REPORT_PATH"], 'w', encoding='utf-8') as f:
        json.dump(history_report, f, ensure_ascii=False, indent=4)
    return history_report

if __name__=="__main__":
    with open(PATH_CONFIG["INPUT_PATH"], 'r', encoding='utf-8') as f:
        raw_info = f.read()
    if_info=saba_if(raw_info)
    saba_qsr(if_info)
    print(f"\n{'*' * 20} 总计消耗 {'*' * 20}")
    print(f"总输入 Token: {TOTAL_USAGE['input']}")
    print(f"总输出 Token: {TOTAL_USAGE['output']}")
    print(f"命中 Token: {TOTAL_USAGE['hit']}")
    print(f"总计 Token: {TOTAL_USAGE['input'] + TOTAL_USAGE['output']}")


