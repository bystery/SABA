# Complex-Narrative-Reasoning: 迭代式框架的复杂叙事推理

## 引用方式 (Citation)：提供 BibTeX 格式的引用信息，方便别人引用你的工作。

## 项目概述

本项目实现了一个基于大语言模型（LLM）的迭代式框架（IF-QSR），用于解决复杂叙事中的故事推理任务。通过问题分解与故事推理的迭代循环，系统能够处理多线索、多角色的复杂叙事场景，并在不同难度级别的数据集上进行评估。

## 📁 项目结构
Complex-Narrative-Reasoning/  
├── ablation_study/          # 消融实验代码与结果  
├── baseline/                # 基线模型实现  
├── dataset/                 # 三个难度等级的数据集  
├── Evaluation_Similarity/     # 评估指标计算  
├── IF-QSR/                  # 主框架实现  
├── sbert-base-chinese-nli/  # 相似度模型（已下载）  
├── .gitignore  
├── config.yaml             # 配置文件   
├── LICENSE  
├── README.md              # 本文档  
└── requirements.txt       # Python依赖  

## 快速开始

步骤1：环境配置
1. 创建虚拟环境（Python 3.8+） python -m venv myenv  
2. 激活虚拟环境
3. 安装依赖  pip install -r requirements.txt

步骤2：配置大模型,编辑 config.yaml 文件：  

（1）大模型配置  
api_config:  
  base_url: "https://api.deepseek.com/v1"    
  api_key_env_var: "DEEPSEEK_API_KEY"  
llm_settings:   
  model_name: "deepseek-chat"  

（2）测试案件配置  
paths:  
  input_data_dir: "../dataset/artice_path"  # 指向具体案件目录  

步骤3：运行主框架 **IF-QSR.py**  
运行成功后，生成两个输出文件：  
SR_model_output.json – 保存最终结果（列表中的最后一个元素），用于召回率计算  
Q_model_output.json – 保存中间结果，用于线索覆盖度计算  

步骤4：评估方法  
召回率计算  
1.从 SR_model_output.json 提取最后一个元素  
2.复制到 Evaluation_Similarity/awaiting_test_results.json  
3.修改with open(r"../dataset/对应案件/gold-standard-answer-atomic.json", 'r', encoding='utf-8') as f:  
4.运行 Evaluation_Similarity/Evaluation.py  

线索覆盖度计算  
1.查看 predefined_cues.txt 中的预定义线索  
2.人工检查 Q_model_output.json 是否提及这些线索  
3.计算提及线索的覆盖率  

## 数据集
dataset 目录中存储三个不同等级难度的数据集：   
1.简单难度 (simple_case/)  
2.中等难度 (medium_case/)  
3.困难难度 (complex_case/)  
**每个案件目录包含：**  
输入文件  
1.Event.json – 事件描述  
2.evidence.json – 证据链  
3.role.json – 角色信息  
评估文件  
1.gold-standard-answer-atomic.json – 黄金标准答案（用于召回率计算）  
2.predefined_cues.txt 案件预定义线索 （用于线索覆盖度计算）

## baseline模型的评估
线索覆盖度评估  
1.运行基线模型，将控制台输出内容保存为 .txt 文件  
2.寻找输出中是否提及 predefined_cues.txt 中的线索  
3.人工进行线索覆盖度的计算  

召回率评估  
1.提取控制台输出的最后结果  
2.复制到 Evaluation_Similarity/awaiting_test_results.json    
3.修改with open(r"../dataset/对应案件/gold-standard-answer-atomic.json", 'r', encoding='utf-8') as f:    
4.运行 Evaluation_Similarity/Evaluation.py    






