markdown
# Hybrid Sparse-Dense RAG

A retrieval-augmented system that fuses sparse (SPLADE) and dense (Contriever) embeddings for improved passage retrieval.

## Project Structure
src/retrieval/
├── splade.py # SPLADE wrapper with splade_embed()
├── dense.py # Contriever wrapper with dense_embed()
├── hybrid_retriever.py # Combines sparse & dense scores via alpha
└── notebooks/
└── demo.ipynb # Interactive usage examples

evaluation/
├── beir_metrics/ # Standardized evaluation
│ └── evaluator.py # ir_measures wrapper
└── evaluation.py # Main evaluation pipeline


## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/spenkov101/hybrid-sparse-dense-RAG.git
   cd hybrid-sparse-dense-RAG
Install dependencies:

bash
pip install -r requirements.txt
pip install ir_measures  # For evaluation metrics
Set Python path:

bash
export PYTHONPATH=$PWD/src  # Linux/Mac
$env:PYTHONPATH = "$PWD/src"  # PowerShell
Usage
Basic Retrieval
python
from retrieval.hybrid_retriever import HybridRetriever
retriever = HybridRetriever()
passages = ["Paris is...", "Berlin is..."]
results = retriever.hybrid_search("French capital?", passages, alpha=0.5)
Evaluation
python
from evaluation import run_beir_evaluation

# qrels = {qid: {docid: relevance}}  # Ground truth
# results = {qid: {docid: score}}    # Your retriever output
metrics = run_beir_evaluation(qrels, results)  # Returns nDCG@10, P@5, Recall@100
Supported Metrics:

nDCG@[k]: Rank-aware accuracy

P@[k]: Precision at k documents

Recall@[k]: Recall at k documents

Future Work
Quantization: Export Contriever to ONNX for low-latency inference

Demo: Build Gradio/Streamlit UI

Expanded Metrics: Add MRR@10 and MAP@100

License
MIT