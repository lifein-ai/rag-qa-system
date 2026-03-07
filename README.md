# RAG (Retrieval-Augmented Generation) 问答系统

## 项目简介

这是一个基于 **DeepSeek** 和 **FAISS** 的 **RAG (Retrieval-Augmented Generation)** 问答系统。它能够根据给定的文档，在检索相关信息后，利用大语言模型（如 DeepSeek）生成问题的答案。该项目支持文档切分、向量化、检索和答案生成，同时可以进行检索结果的溯源。

## 技术栈

- **Python**：编程语言
- **Sentence Transformers**：用于生成段落的向量表示
- **FAISS**：高效的相似度检索库，用于构建文档的向量索引
- **DeepSeek**：通过 OpenAI API 提供的自然语言处理服务，用于生成答案
- **dotenv**：用于从 `.env` 文件加载环境变量
- **logging**：用于记录程序日志

## 项目结构

rag_project/
├─ app.py                 # 主程序，用户交互入口
├─ rag_core.py            # 核心问答流程，包括文档处理、检索和答案生成
├─ llm_client.py          # 与 DeepSeek API 的交互
├─ storage.py             # 保存和加载 FAISS 索引、段落数据
├─ .env                   # API 密钥文件
├─ data/
│  ├─ faiss_indexes/      # 存储 FAISS 索引文件
│  └─ paragraphs/         # 存储切分后的段落
└─ eval/
   ├─ eval.jsonl          # 评测集文件
   ├─ run_eval.py         # 用于评测的脚本
   └─ eval_results.csv    # 评测结果保存文件

## 功能

1. **文档处理**：
   - 加载文档并将其切分为多个段落。
   - 每个段落都被向量化，便于后续检索。

2. **检索**：
   - 使用 FAISS 索引进行高效检索，查找与用户问题最相关的文档段落。

3. **答案生成**：
   - 使用 DeepSeek（通过 OpenAI API）生成基于检索结果的答案。

4. **空检索处理**：
   - 当检索不到相关内容时，系统会返回一个友好的提示，建议用户换一种问题表达。

5. **来源溯源**：
   - 每个检索到的段落都会输出其来源（文档路径和段落编号）。

## 安装与使用

### 1. 克隆项目

git clone <项目仓库链接>  
cd rag_project

### 2. 创建虚拟环境并安装依赖

首先，创建一个虚拟环境：

python -m venv venv

然后，激活虚拟环境并安装依赖：

- Windows:

venv\Scripts\activate

- macOS/Linux:

source venv/bin/activate

接着，安装所需的 Python 库：

pip install -r requirements.txt

### 3. 配置 API 密钥

在项目根目录下创建一个 `.env` 文件，填入你的 OpenAI API 密钥：

OPENAI_API_KEY=your-api-key

### 4. 运行程序

启动问答系统：

python app.py

### 5. 文档输入

在运行程序时，系统会提示你输入要进行评测的文档路径。输入完整路径后，程序会加载文档并启动问答功能。

## 示例输出

### 查询示例

请输入你的问题：RAG的核心思想是什么？  
=== ANSWER ===  
根据下文，RAG是一种技术，但具体定义未提供。  

=== SOURCES ===  
chunk_id=1, source=D:\RAG\Text.txt, distance=0.5180  
chunk_id=26, source=D:\RAG\Text.txt, distance=0.5491  
chunk_id=14, source=D:\RAG\Text.txt, distance=0.5606

## 贡献

如果你有任何改进或问题，欢迎提交 Pull Request 或 Issues。

## 致谢

感谢以下项目和库的支持：

- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [DeepSeek](https://deepseek.com/)
- [OpenAI](https://openai.com/)