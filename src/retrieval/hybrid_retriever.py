from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import SentenceTransformer  # <-- Add this import
import numpy as np
from typing import List
import torch

class HybridRetriever:
    def __init__(self):
        # SPLADE model (sparse lexical)
        self.splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
        self.splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
        
        # Dense model (semantic)
        self.dense_model = SentenceTransformer("facebook/contriever")

    def splade_embed(self, text: str) -> torch.Tensor:
        """Generate SPLADE sparse embeddings"""
        inputs = self.splade_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.splade_model(**inputs).logits
        return torch.max(torch.log(1 + torch.relu(logits)), dim=1)[0].squeeze()

    def dense_embed(self, text: str) -> np.ndarray:
        """Generate dense embeddings"""
        return self.dense_model.encode(text)
    
    def hybrid_search(self, query: str, passages: List[str], alpha: float = 0.5) -> List[str]:
        """Combine sparse and dense scores"""
        sparse_embs = torch.stack([self.splade_embed(p) for p in passages])
        dense_embs = np.stack([self.dense_embed(p) for p in passages])
        
        # Sparse similarity (dot product)
        query_sparse = self.splade_embed(query)
        sparse_scores = torch.matmul(sparse_embs, query_sparse)
        
        # Dense similarity (cosine)
        query_dense = self.dense_embed(query)
        dense_scores = np.dot(dense_embs, query_dense) / (
            np.linalg.norm(dense_embs, axis=1) * np.linalg.norm(query_dense))
        
        # Hybrid score
        combined = alpha * sparse_scores + (1-alpha) * dense_scores
        return [passages[i] for i in combined.argsort(descending=True)]

if __name__ == "__main__":
    retriever = HybridRetriever()
    passages = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "The Eiffel Tower is in Paris"
]
results = retriever.hybrid_search("What is the French capital?", passages)
print("Top passage:", results[0])
    # query = "What is the capital of France?"
    # print("SPLADE embedding shape:", retriever.splade_embed(query).shape)
    # print("Dense embedding shape:", retriever.dense_embed(query).shape)
    