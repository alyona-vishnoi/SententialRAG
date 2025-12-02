# SentenialRAG  
### A Lightweight, Self-Correcting Multi-Agent RAG System

SentenialRAG is a multi-agent Retrieval-Augmented Generation (RAG) system designed to produce **accurate, cite-verified answers** for technical and research-heavy queries. The system focuses on correctness, traceability, and automated quality checks—avoiding the silent failures common in traditional RAG pipelines.

> **Coming soon:** Evaluation experiments and visualizations!!!

---

## 🚀 Overview

SentenialRAG uses a small set of specialized agents to improve reliability at each step of the pipeline:

1. **Query Analyzer** – Structures and refines the user’s question  
2. **Hybrid Retriever** – Combines vector search + BM25  
3. **Retrieval Evaluator** – Ensures the retrieved context is relevant  
4. **Answer Generator** – Produces grounded answers with citations  
5. **Fact Checker** – Verifies each claim against source documents  
6. **Quality Scorer** – Scores factuality, relevance, citation correctness, and clarity

The goal is to build a **robust, low-cost, local-first RAG system** that catches its own mistakes and flags uncertain answers.

---

## 🔑 Key Features

- **Hybrid Retrieval (Vector + BM25)** for stronger coverage  
- **Smarter chunking tuned for research papers**  
- **Citation generation + validation**  
- **Automated fact-checking of each claim**  
- **Quality scoring with PASS / FLAG / REGENERATE logic**

---

## 🛠️ Tech Stack

- Python 3.9+  
- LangGraph for agent orchestration  
- LangChain for retrieval + pipelines  
- ChromaDB for vector storage  
- HuggingFace Sentence-Transformers (MiniLM-L6-v2)  
- BM25 (rank-bm25)  
- Google Gemini 1.5 Flash (LLM)

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/sentenialrag
cd sentenialrag

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_api_key" > .env
```
