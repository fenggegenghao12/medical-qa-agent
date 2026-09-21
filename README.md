# 医疗问诊多模态 RAG + Agent 系统

基于自主 Agent 循环 + 混合检索 RAG + 多模态图片匹配的医疗问诊 Demo。

用户提问后，Agent 自主决策是否调用工具：先用本地医疗知识库回答问题，再用 CLIP 检索匹配图片并由 VLM 二次裁决选图，最后可登记患者信息完成预约。

## 架构流程

```
用户提问
  │
  ├─ 文本路：查询改写×3 → BM25+向量混合检索 → RRF融合 → 答案重选
  │
  └─ 图片路：主题提取 → CLIP文搜图(5张) → VLM裁决(选1张) → 展示
  │
  └─ Agent自主调用工具：query_answer / get_query_time / record_patient_information
```

## 技术栈

| 模块 | 技术 |
| --- | --- |
| Agent 循环 | LangChain ChatOpenAI + bind_tools + 手写 invoke 循环 |
| 文本检索 | BM25 (jieba分词) + Chroma向量库 (BGE-large-zh) + RRF融合 |
| 图片检索 | Chinese-CLIP (ViT-B-16) + FAISS IndexFlatIP |
| 多模态裁决 | qwen3.8-27b VLM (图片以 base64 Data URL 传入) |
| 工具定义 | LangChain @tool + Pydantic BaseModel 参数约束 |
| 会话隔离 | SessionManager 按 session_id 管理独立 Agent 实例 |
| 配置管理 | pydantic-settings 读取 .env |

## 项目结构

```
backend/
├── main.py                 # Agent循环 + SessionManager会话隔离
├── tools.py                # 3个工具定义(RAG问答/获取时间/记录患者)
├── config.py               # pydantic-settings读取.env
├── SKILL.md                # 技能文档(system prompt + 工具清单)
├── requirements.txt        # 依赖清单
├── .env.example            # 环境变量模板
└── rags/medical_qa/
    ├── table_rags.py       # 核心管线：文本RAG + CLIP图片 + VLM选图
    ├── download_img.py     # 主题图片批量下载工具
    ├── generate_train_data.py  # 微调数据抽样清洗
    └── Chinese-medical-dialogue/
        └── data/
            └── medical_100.json  # 示例数据(100条)
```

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/fenggegenghao12/medical-qa-agent.git
cd medical-qa-agent

# 创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API

```bash
# 复制环境变量模板
cp .env.example .env
# 编辑 .env，填入你自己的 API Key
```

需要配置：
- 阿里云百炼/DashScope API Key（用于 qwen 大模型）
- Ollama 本地服务（用于 BGE-large-zh 向量化，需安装 Ollama 并拉取模型）

### 3. 初始化知识库

首次运行需要建库（在 `rags/medical_qa/` 目录下）：

```python
from table_rags import json_rags, generate_retriever, generate_vectors, build_faiss_index

# 建文本向量库
docs = json_rags()
generate_retriever(docs)

# 建图片向量索引
vectors = generate_vectors()
build_faiss_index(vectors)
```

### 4. 运行

```bash
python main.py
```

## 核心设计亮点

1. **自主 Agent 循环**：手写 invoke → tool_calls → ToolMessage 循环，非框架封装，可完全控制流程
2. **invalid_tool_calls 容错**：模型输出非法 JSON 参数时，自动解析修复后重新执行工具
3. **混合检索 + RRF**：BM25(关键词) + 向量(语义) 两路检索，权重 [0.2, 0.8]，RRF(k=60) 融合
4. **多查询改写**：LLM 将用户问题扩写为 3 句同义问，分别检索后融合，提升召回率
5. **多模态漏斗设计**：CLIP 粗召回(top-5) → VLM 精排(选1)，兼顾效率与准确率
6. **Pydantic 工具约束**：用 BaseModel 显式声明参数 Schema，让模型知道要填什么字段
7. **会话隔离**：SessionManager 按 session_id 存取独立 Agent 实例，不同用户对话互不串话

## 工具说明

| 工具 | 功能 | 调用时机 |
| --- | --- | --- |
| query_answer | 基于RAG知识库回答医疗问题 | 用户咨询症状/疾病/用药 |
| get_query_time | 获取当前咨询时间 | 记录患者信息前调用 |
| record_patient_information | 记录患者信息到CSV | 患者信息齐全且需要预约时 |

## 数据来源

- 医疗问答数据：[Chinese-medical-dialogue](https://huggingface.co/datasets/)（80万条，本项目抽样500条建库）
- 主题图片：通过 bing_image_downloader 按疾病主题批量下载
