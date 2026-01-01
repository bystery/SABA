# SABA (Self-Awareness Before Action) 官方实现仓库

本仓库是论文 《Self-Awareness before Action: Stop and Identify Logical Gaps in Complex Reasoning》 的官方实现代码。

# 📖 项目简介

SABA 是一种创新的通用推理框架。与传统的“生成后再修正”（如 Self-Refine）不同，SABA 引入了 前置觉察（Self-Awareness） 机制。该机制使大语言模型能够在执行推理动作前，主动识别任务逻辑链中的复杂程度与信息断层。在复杂叙事推理任务中，显著减少“逻辑跳跃”并有效抑制幻觉生成。

# 🌟 本仓库包含
- 自建的```Detective Puzzle```数据集✅
- SABA实现✅

# 📁 项目结构
```
SABA/  
├── SABA/                    # SABA 主框架实现  
├── dataset/                 # 包含 Simple, Medium, Complex 三个等级的案件集  
├── Evaluation_Similarity/   # 评估指标计算模块  
├── requirements.txt         # 环境依赖列表  
├── LICENSE                  # 项目许可证  
└── README.md                # 本说明文档  
```

# 🚀 快速开始

## 1. 环境配置

建议使用 Python 3.12+ 环境：
```
## 创建并激活虚拟环境
python -m venv myenv
source myenv/bin/activate  # Linux/macOS
## Windows 用户请使用: myenv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 2. 模型与数据配置
直接在代码头部修改配置部分。示例如下：
```
# --- 路径配置 ---
PATH_CONFIG = {
    "MODEL_ENCODER_PATH": "YOUR_MODEL_PATH",
    "GOLD_PATH": "CASE/Answer.txt",
    "TEST_PATH": "CASE/report.json"
}

PATH_CONFIG={
    "INPUT_PATH": "CASE/Mystery_text.txt",
    "OUTPUT_QA_PATH": "CASE/SABA_PostRun_KB/q_a.json",
    "OUTPUT_REPORT_PATH": "CASE/SABA_PostRun_KB/report.json"
}

# --- LLM 客户端配置 ---
LLM_CONFIG = {
    "api_key": "YOUR_API_KEY",
    "base_url": "BASE_URL",
    "model_name": "MODE_NAME",
    "temperature": 0.0
}
```
## 3. 运行主框架
确认配置无误后，执行主程序开始推理过程： 
```
python SABA/SABA.py
```
运行完成后，系统将在对应目录下生成两个关键文件：
- ```q_a.json```: 保存中间探索过程（用于计算线索覆盖度）。  

- ```report.json```: 保存最终推理结果（用于计算召回率）。


# 📊 评估方法
本项目采用双重指标评估模型在复杂叙事推理中的表现：

## 指标 A：Semantic Recall Rate(SRR)，评估模型输出与标准答案的匹配程度。
- 下载```sbert-base-chinese-nli```模型
- 设置路径配置运行评估脚本
```
python Evaluation_Similarity/Evaluation.py
```

## 指标 B：ClueCoverageRate(CCR)，评估模型是否识别并利用了案件的关键线索。
- 查看对应案件目录下的 ```predefined_cues.txt```。
- 人工检查```q_a.json```中是否包含了上述预定义线索。

# 📂 数据集说明

dataset 目录按难度分为三个等级，每个案件目录包含：
- 输入端: ```Mystery_text.txt```原始案件信息
- SABA输出: ```PostRun_KB```目录包含了SABA输出
- 基准端: ```predefined_cues.txt```包含预定义线索、```Answer.txt```包含黄金结果

# 📜 引用方式
如果您在研究中使用了本仓库的代码或 SABA 框架，请引用我们的论文：
```

```