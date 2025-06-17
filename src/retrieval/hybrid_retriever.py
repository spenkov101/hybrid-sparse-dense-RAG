from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class HybridRetriever:
    def __init__(self):
        self.splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.dense_model = SentenceTransformer("facebook/contriever")

    def splade_embed(self, text: str) -> torch.Tensor:
        inputs = self.splade_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.splade_model(**inputs).logits
        return torch.max(torch.log(1 + torch.relu(logits)), dim=1)[0].squeeze()

    def dense_embed(self, text: str) -> np.ndarray:
        return self.dense_model.encode(text)
    
    def hybrid_search(self, query: str, passages: List[str], alpha: float = 0.5) -> List[dict]:
        sparse_embs = torch.stack([self.splade_embed(p) for p in passages])
        dense_embs = np.stack([self.dense_embed(p) for p in passages])
        
        query_sparse = self.splade_embed(query)
        sparse_scores = torch.matmul(sparse_embs, query_sparse).numpy()
        
        query_dense = self.dense_embed(query)
        dense_scores = np.dot(dense_embs, query_dense) / (
            np.linalg.norm(dense_embs, axis=1) * np.linalg.norm(query_dense))
        
        combined = alpha * sparse_scores + (1-alpha) * dense_scores
        ranked = combined.argsort()[::-1]
        
        return [{
            "text": passages[i],
            "score": float(combined[i]),
            "sparse_score": float(sparse_scores[i]),
            "dense_score": float(dense_scores[i])
        } for i in ranked]