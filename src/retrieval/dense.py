from sentence_transformers import SentenceTransformer
import numpy as np

class DenseRetriever:
    def __init__(self):
        # Load Contriever for dense embeddings
        self.dense_model = SentenceTransformer("facebook/contriever")
    def dense_embed(self, text: str) -> np.ndarray:
        """Generate dense embeddings"""
        return self.dense_model.encode(text)