import os
from typing import List

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings


class HFInferenceEmbeddings(Embeddings):
    """Embeddings via HuggingFace Inference API - no local weights needed."""

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        api_key: str | None = None,
    ):
        self.model = model
        self._client = InferenceClient(
            api_key=api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._client.feature_extraction(texts, model=self.model)
        if hasattr(result, "tolist"):
            result = result.tolist()
        if not isinstance(result, list):
            result = list(result)
        if len(result) > 0 and not isinstance(result[0], list):
            result = [result]
        return [list(map(float, vec)) for vec in result]

    def embed_query(self, text: str) -> List[float]:
        result = self._client.feature_extraction(text, model=self.model)
        if hasattr(result, "tolist"):
            result = result.tolist()
        if not isinstance(result, list):
            result = list(result)
        if len(result) > 0 and isinstance(result[0], list):
            result = result[0]
        return list(map(float, result))