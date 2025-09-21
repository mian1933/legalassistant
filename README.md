## 📥 Download Models

We use two types of models in this project. Please download them:

- **Language Model**: [Qwen/Qwen-14B-Chat](https://www.modelscope.cn/models/Qwen/Qwen-14B-Chat).
- **Embedding Model**: [bge-m3](https://huggingface.co/BAAI/bge-m3).

---

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

