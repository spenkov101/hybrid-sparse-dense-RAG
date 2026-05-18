# 🔍 Hybrid Sparse–Dense RAG

**A retrieval-augmented system that fuses sparse (SPLADE) and dense (Contriever) embeddings**  
to improve relevance and robustness in passage retrieval.

This project demonstrates **hybrid retrieval scoring** where:
- SPLADE → captures lexical relevance (exact terms, sparse vectors)
- Contriever → captures semantic similarity (dense vectors)
- Hybrid fusion → balances precision & semantic generalization

> Goal: Practical experiments toward **hybrid ranking** in RAG pipelines.

---
## 📐 Architecture

A high-level overview of the hybrid sparse–dense retrieval pipeline  
(including SPLADE, Contriever, and fusion logic):

➡️ **[Architecture Overview](docs/architecture.md)**

## 📂 Project Structure

```
src/retrieval/
├── splade.py            # SPLADE wrapper (sparse_embed)
├── dense.py             # Contriever wrapper (dense_embed)
├── hybrid_retriever.py  # Weighted fusion of sparse+dense scores
└── example_run.py       # Minimal demo script

notebooks/
└── demo.ipynb           # Interactive usage examples

README.md
requirements.txt
```

---

## 🚀 Quick Start

Clone & install:

```bash
git clone https://github.com/spenkov101/hybrid-sparse-dense-RAG.git
cd hybrid-sparse-dense-RAG
pip install -r requirements.txt
```

Ensure `src` is visible to Python:

```bash
export PYTHONPATH=$PWD/src
```

Windows PowerShell:
```powershell
$env:PYTHONPATH = "$PWD/src"
```

---

## ▶️ Minimal Example

Run the test script:

```bash
python src/retrieval/example_run.py
```

Example output:

```
Query: What is the French capital?
Top-1: Paris is the capital of France  (score=...)
```

---

## 🧠 Usage in Code

### SPLADE Embeddings

```python
from retrieval.splade import SpladeRetriever
retriever = SpladeRetriever()
emb = retriever.sparse_embed("Paris is the capital of France")
```

### Dense Embeddings (Contriever)

```python
from retrieval.dense import DenseRetriever
retriever = DenseRetriever()
emb = retriever.dense_embed("Paris is the capital of France")
```

### Hybrid Search

```python
from retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever(alpha=0.5)

passages = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
]

results = retriever.search("What is the French capital?", passages)
print(results)
```

---

---

## 📊 Evaluation

This project includes **BEIR-style evaluation utilities** based on
[`ir_measures`](https://github.com/terrierteam/ir_measures).

The evaluation pipeline supports standard IR metrics such as:

- nDCG@k
- MAP@k
- Precision@k
- Recall@k

Evaluation helpers are implemented in:

src/evaluation/
├── evaluation.py
└── beir_metrics/
└── evaluator.py

This allows direct comparison between **sparse**, **dense**, and **hybrid**
retrieval strategies on benchmark datasets.


## 🧪 Coming Soon

| Feature | Status |
|--------|:-----:|
| BEIR-based evaluation (MAP / nDCG / Recall) | ⏳ |
| ONNX export + quantization | ⏳ |
| Gradio/Streamlit UI demo | ⏳ |

---
## Quick Retrieval Sanity Check

A lightweight notebook is available for inspecting hybrid retrieval outputs:

```bash
notebooks/retrieval_sanity_check.ipynb
```
## 📜 License

MIT License
