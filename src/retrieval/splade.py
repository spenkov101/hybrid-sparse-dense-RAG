from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class SpladeRetriever:
    def __init__(self):
        # Initialize SPLADE model and tokenizer
        self.splade_tokenizer = AutoTokenizer.from_pretrained(
            "naver/splade-cocondenser-ensembledistil"
        )
        self.splade_model = AutoModelForMaskedLM.from_pretrained(
            "naver/splade-cocondenser-ensembledistil"
        )


    def splade_embed(self, text: str) -> torch.Tensor:
        """Generate SPLADE sparse embeddings"""
        inputs = self.splade_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.splade_model(**inputs).logits
        # Apply ReLU + log1p then take max over the sequence dimension
        return torch.max(torch.log(1 + torch.relu(logits)), dim=1)[0].squeeze()
    
