"""Small orchestration helpers used by the Streamlit app and demo script."""

from __future__ import annotations

from .extraction import extract_from_chunks
from .llm import LLMClient
from .models import DocumentChunk, KnowledgeExtraction
from .retrieval import Retriever


def answer_question(retriever: Retriever, question: str, llm: LLMClient, top_k: int = 4) -> tuple[str, list]:
    hits = retriever.search(question, top_k=top_k)
    if not hits:
        return "No relevant passages were retrieved. Try rephrasing the question or adding documents.", []
    try:
        answer = llm.answer(question, [(hit.chunk, hit.score) for hit in hits])
        mode = "LLM-generated answer grounded in retrieved passages"
    except Exception as exc:
        # Fallback is deliberately extractive and does not pretend an LLM was used.
        answer = "LLM unavailable; showing retrieved evidence instead.\n\n" + "\n\n".join(
            f"**{hit.chunk.source}" + (f" — page {hit.chunk.page}" if hit.chunk.page else "")
            + f"** (retrieval score {hit.score:.3f})\n\n{hit.chunk.text}"
            for hit in hits
        )
        mode = f"retrieval-only fallback ({type(exc).__name__})"
    return answer, [(hit.chunk, hit.score) for hit in hits] + [(mode, 0.0)]


def extract_knowledge(chunks: list[DocumentChunk], llm: LLMClient) -> tuple[KnowledgeExtraction, str]:
    try:
        extraction = llm.extract(chunks)
        return extraction, "LLM-assisted structured extraction"
    except Exception as exc:
        return extract_from_chunks(chunks), f"vocabulary-based fallback ({type(exc).__name__})"
