# SentenialRAG  
### A Lightweight, Self-Correcting Multi-Agent RAG System

**Built to catch what other RAG systems miss:** hallucinations, missing citations, and unreliable answers.

SentenialRAG is a multi-agent Retrieval-Augmented Generation (RAG) system designed to produce **accurate, cite-verified answers** for technical and research-heavy queries. The system focuses on correctness, traceability, and automated quality checks—avoiding the silent failures common in traditional RAG pipelines.

---
## 🎯 The Problem

Traditional RAG systems fail silently:
- ❌ **Hallucinate facts** even with correct source documents
- ❌ **No citation tracking** - users can't verify claims
- ❌ **Silent quality degradation** - systems don't know when they're wrong
- ❌ **Binary responses** - either answer or fail, no nuance

**Result:** Users can't trust the outputs, limiting real-world deployment.
---

## 🚀 The Solution

SentenialRAG uses a small set of specialized agents to improve reliability at each step of the pipeline:

1. **Query Analyzer** – Structures and refines the user’s question  
2. **Hybrid Retriever** – Combines vector search + BM25  
3. **Retrieval Evaluator** – Ensures the retrieved context is relevant  
4. **Answer Generator** – Produces grounded answers with citations  
5. **Fact Checker** – Verifies each claim against source documents  
6. **Quality Scorer** – Scores factuality, relevance, citation correctness, and clarity

**Key Innovation:** The system can assess its own reliability and flag uncertain outputs rather than confidently returning bad answers.

---
## Real Results
Tested on **27 ML research papers** (GPT-3, BERT, Transformers, Vision Transformers, + 20 recent papers):

### Example:
**Query:** *"What is the architecture of BERT?"*
```
✅ Answer (Score: 0.90 - GOOD)

BERT uses a bidirectional Transformer [1]. Its representations 
are jointly conditioned on both left and right context in all 
layers [1].

📊 Quality Metrics:
   Factual Accuracy:  1.00 (100%)
   Citation Quality:  1.00 (100%)
   Relevance:         1.00 (100%)
   Completeness:      0.50 (50%)

🔍 Fact Verification:
   Claims verified:   2/2 (100%)
   Avg confidence:    100%
   
✅ Recommendation: PASS

💡 System Feedback:
"Answer is accurate but could include more architectural details 
(layers, attention heads, hidden dimensions)."
```
**Demonstrates:**
- 100% factual accuracy with full verification
- Self-aware about completeness limitations
- Still passes quality threshold (0.90 > 0.8)
---
## 📊 System Performance Summary

| Metric | Value | Insight |
|--------|-------|---------|
| **Best Score** | 0.99 | Near-perfect on factual queries |
| **Avg Fact Accuracy** | 0.85+ | High correctness when answering |
| **Citation Rate** | 100% | Every answer includes sources |
| **Claim Verification** | 83%+ | Most claims verified against sources |
| **Self-Awareness** | ✅ | Flags unreliable outputs (score 0.20) |
---

## 🔑 Key Features

- **Hybrid Retrieval (Vector + BM25)** for stronger coverage  
- **Smarter chunking tuned for research papers**  
- **Citation generation + validation**  
- **Automated fact-checking of each claim**
  - **Example:**
      ```python
      Claim: "GPT-3 has 175 billion parameters"
      Evidence: "GPT-3 175B ... 174,600 ..." (from source)
      Verified: ✅ YES
      Confidence: 1.0
      ```
- **Multi-Dimensional Quality scoring with PASS / FLAG / REGENERATE logic**
  - Automatically retries when retrieval quality is low.

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

