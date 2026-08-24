# 🔍 Hybrid Sparse–Dense RAG

**A retrieval-augmented system that fuses sparse (SPLADE) and dense (Contriever) embeddings**  
to improve relevance and robustness in passage retrieval.

This project demonstrates **hybrid retrieval scoring** where:

- SPLADE → captures lexical relevance through exact terms and sparse vectors
- Contriever → captures semantic similarity through dense vectors
- Hybrid fusion → balances lexical precision and semantic generalization

> Goal: Practical experiments toward **hybrid ranking** in RAG pipelines.

---

## 📐 Architecture

A high-level overview of the hybrid sparse–dense retrieval pipeline, including SPLADE, Contriever, and fusion logic:

➡️ **[Architecture Overview](docs/architecture.md)**

## 📂 Project Structure

```text
src/retrieval/
├── splade.py            # SPLADE wrapper
├── dense.py             # Contriever wrapper
├── hybrid_retriever.py  # Weighted sparse-dense fusion
└── example_run.py       # Minimal demo script

notebooks/
└── retrieval_sanity_check.ipynb

README.md
requirements.txt
```

---

## 🚀 Quick Start

Clone the repository and install its dependencies:

```bash
git clone https://github.com/spenkov101/hybrid-sparse-dense-RAG.git
cd hybrid-sparse-dense-RAG
pip install -r requirements.txt
```

Ensure the repository root is visible to Python:

```bash
export PYTHONPATH=$PWD
```

Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD"
```

---

## ▶️ Minimal Example

Run the example script:

```bash
python src/retrieval/example_run.py
```

Example output:

```text
Query: What is the French capital?
Top-1: Paris is the capital of France (score=...)
```

---

## 🧠 Usage in Code

### SPLADE Embeddings

```python
from src.retrieval import SpladeRetriever

retriever = SpladeRetriever()
embedding = retriever.sparse_embed(
    "Paris is the capital of France."
)
```

### Dense Embeddings with Contriever

```python
from src.retrieval import DenseRetriever

retriever = DenseRetriever()
embedding = retriever.dense_embed(
    "Paris is the capital of France."
)
```

### Hybrid Search

```python
from src.retrieval import (
    HybridRetriever,
    create_search_config,
    get_top_result,
)

passages = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
]

config = create_search_config(alpha=0.5, top_k=2)
retriever = HybridRetriever()

results = retriever.hybrid_search(
    "What is the French capital?",
    passages,
    alpha=config["alpha"],
    top_k=config["top_k"],
)

top_result = get_top_result(results)

if top_result:
    print(top_result["text"])
    print(top_result["score"])
```

---

## 📊 Evaluation

This project includes **BEIR-style evaluation utilities** based on
[`ir_measures`](https://github.com/terrierteam/ir_measures).

The evaluation pipeline supports standard information-retrieval metrics such as:

- nDCG@k
- MAP@k
- Precision@k
- Recall@k

Evaluation helpers are implemented in:

```text
evaluation/
├── evaluation.py
```

These utilities support comparisons between **sparse**, **dense**, and **hybrid** retrieval strategies on benchmark datasets.

---

## 🧪 Coming Soon

| Feature | Status |
|---|:---:|
| ONNX export and quantization | ⏳ |
| Gradio or Streamlit demo | ⏳ |

---

## Quick Retrieval Sanity Check

A lightweight notebook is available for inspecting hybrid retrieval outputs:

```text
notebooks/retrieval_sanity_check.ipynb
```

---

## 📜 License

MIT License