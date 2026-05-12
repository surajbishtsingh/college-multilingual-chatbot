# rag/embeddings.py
# Single place for embedding model — imported everywhere
import os
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_embed_model = None

# Lightweight multilingual model — supports Hindi + English (and 50+ languages)
# 6 layers, ~120MB, fast on CPU — lightest option with strong Hindi/English quality
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embed_model() -> HuggingFaceEmbeddings:
    """
    Singleton embedding model.
    Loaded once, reused everywhere.
    Saves ~120MB RAM by not loading multiple times.
    """
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32,  # increased — smaller model handles larger batches easily
            },
        )
        print("[Embed] Model loaded")
    return _embed_model


def embed_text(text: str) -> list[float]:
    """Embed a single string → vector."""
    return get_embed_model().embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple strings → list of vectors."""
    return get_embed_model().embed_documents(texts)
