# Hybrid Sparse-Dense RAG

A proof-of-concept retrieval-augmented system that fuses sparse (SPLADE) and dense (Contriever) embeddings for improved passage retrieval.

## Project Structure

```
src/retrieval/
├── splade.py           # SPLADE wrapper with splade_embed()
├── dense.py            # Contriever wrapper with dense_embed()
├── hybrid_retriever.py # Combines sparse & dense scores via alpha
└── notebooks/
    └── demo.ipynb      # Interactive usage examples

README.md
requirements.txt
```

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/spenkov101/hybrid-sparse-dense-RAG.git
   cd hybrid-sparse-dense-RAG
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Ensure `src` is on your PYTHONPATH:

   ```bash
   export PYTHONPATH=$PWD/src
   ```

   *(Windows PowerShell: `$env:PYTHONPATH = "$PWD/src"`)*

## Usage

### SPLADE Embeddings

```python
from retrieval.splade import SpladeRetriever
retriever = SpladeRetriever()
emb = retriever.splade_embed("Paris is the capital of France")
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
retriever = HybridRetriever()
passages = ["Paris is the capital of France", "Berlin is the capital of Germany"]
results = retriever.hybrid_search("What is the French capital?", passages, alpha=0.5)
print(results)
```

## Future Work

* **Evaluation**: Add BEIR-based evaluation scripts using `ir_measures` or `pytrec_eval` to compare sparse, dense, and hybrid performance.
* **Quantization**: Export Contriever to ONNX and quantize for low-latency inference.
* **Demo**: Build a Gradio/Streamlit UI for interactive querying.

## License

MIT