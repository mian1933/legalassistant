# ⚖️ Legal Assistant
**Legal Assistant** is an AI-powered system designed to assist with **legal document analysis** and **question answering**.  It combines **large language models (LLMs)** with **retrieval-augmented generation (RAG)**, enabling accurate and efficient responses across a wide range of legal queries.

## ⚙️ Setup Environment

First, create a virtual environment and install all dependencies:

```bash
pip install -r requirements.txt
```

## 📥 Download Models

We use two types of models in this project. Please download them:

- **Language Model**: [Qwen/Qwen-14B-Chat](https://www.modelscope.cn/models/Qwen/Qwen-14B-Chat).
- **Embedding Model**: [bge-m3](https://huggingface.co/BAAI/bge-m3).

---
## 📥 Download datasets

- [dataset1](https://huggingface.co/datasets/reglab/barexam_qa)  
- [dataset2](https://huggingface.co/datasets/reglab/housing_qa)  

## 🏗️ Build Index

To enable efficient retrieval, we need to build a vector database:

```bash
pip install chromadb

python embedding2.py
```

## ❓ **Run Question Answering**

Once the index is built, you can run the QA pipeline with:

```bash
python run.py

