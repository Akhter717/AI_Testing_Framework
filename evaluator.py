"""RAGAS evaluation system: a SEPARATE Groq judge model + the same embedding model."""
# Compatibility patch: ragas 0.2.x imports langchain_community's Vertex AI classes,
# which newer langchain-community versions no longer ship. Register harmless stand-ins
# BEFORE ragas is imported.
import sys
import types

_m = types.ModuleType("langchain_community.chat_models.vertexai")
_m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _m
_m = types.ModuleType("langchain_community.llms.vertexai")
_m.VertexAI = type("VertexAI", (), {})
sys.modules["langchain_community.llms.vertexai"] = _m

import asyncio
import json
import random
import re
from typing import Callable, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

try:
    from ragas.metrics import SemanticSimilarity
except Exception:  # not available in every ragas version
    SemanticSimilarity = None

import rag

NO_ANSWER = "INSUFFICIENT_CONTEXT"


# ------------------------------------------------------- judge LLM utilities
def make_question(chunks: List[rag.Chunk]) -> str:
    """Pick a random chunk and let the judge LLM write one question about it."""
    pool = [c for c in chunks if len(c.text) > 300] or chunks
    chunk = random.choice(pool)
    q = rag.chat(
        rag.judge_model(),
        "You write exam-style questions.",
        "Write ONE clear, specific question that can be answered using only the passage below. "
        "Do not mention 'the passage' or 'the text'. Output only the question.\n\n"
        f"Passage:\n{chunk.text}",
        temperature=0.7,
    )
    return q.strip().strip('"')


def generate_reference(question: str, chunks: List[rag.Chunk]) -> str:
    """Judge LLM writes its own answer from a WIDER set of chunks."""
    out = rag.chat(
        rag.judge_model(),
        "You are an expert reader. Answer strictly from the context.",
        f"Context:\n{rag.format_context(chunks)}\n\nQuestion: {question}\n\n"
        "Write a complete, concise answer using ONLY the context. "
        f"If the context does not contain the answer, reply exactly: {NO_ANSWER}",
    )
    return out


def check_claims(answer: str, chunks: List[rag.Chunk]) -> List[Dict]:
    """Split the answer into claims and verify each against the retrieved chunks."""
    prompt = (
        "You are a strict fact checker.\n\n"
        f"Context:\n{rag.format_context(chunks)}\n\nAnswer to check:\n{answer}\n\n"
        "Split the answer into atomic factual claims. For each claim decide whether it is FULLY "
        "supported by the context. Reply with ONLY a JSON array, no other text. Each item: "
        '{"claim": "...", "verdict": "supported" or "not_supported", '
        '"evidence": "short quote from the context, or empty"}'
    )
    out = rag.chat(rag.judge_model(), "You output only valid JSON.", prompt)
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        raise ValueError("Judge did not return a JSON list.")
    data = json.loads(m.group(0))
    return [
        {"claim": str(d.get("claim", "")),
         "verdict": "supported" if str(d.get("verdict", "")).lower().startswith("supported") else "not_supported",
         "evidence": str(d.get("evidence", ""))}
        for d in data if isinstance(d, dict)
    ]


# ------------------------------------------------------------- RAGAS metrics
class FastEmbedEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return rag.embed_texts(list(texts)).tolist()

    def embed_query(self, text):
        return rag.embed_texts([text])[0].tolist()


def _build():
    judge = LangchainLLMWrapper(
        ChatGroq(model=rag.judge_model(), temperature=0, max_retries=5)
    )
    emb = LangchainEmbeddingsWrapper(FastEmbedEmbeddings())
    return judge, emb


async def _score(metric, sample):
    try:
        return float(await metric.single_turn_ascore(sample))
    except Exception as e:  # keep going if one metric fails (rate limit / bad JSON)
        return f"ERROR: {type(e).__name__}: {str(e)[:150]}"


async def _run(question, answer, contexts, reference, on_progress):
    judge, emb = _build()
    sample = SingleTurnSample(
        user_input=question, response=answer,
        retrieved_contexts=contexts, reference=reference or None,
    )
    # Groq does not support n>1 completions -> strictness=1
    metrics = {
        "Faithfulness": Faithfulness(llm=judge),
        "Answer relevancy": ResponseRelevancy(llm=judge, embeddings=emb, strictness=1),
        "Context precision (no reference)": LLMContextPrecisionWithoutReference(llm=judge),
    }
    if reference:
        metrics["Context precision (with reference)"] = LLMContextPrecisionWithReference(llm=judge)
        metrics["Context recall"] = LLMContextRecall(llm=judge)
        metrics["Factual correctness"] = FactualCorrectness(llm=judge)
        if SemanticSimilarity is not None:
            metrics["Semantic similarity"] = SemanticSimilarity(embeddings=emb)
    results: Dict[str, object] = {}
    for name, metric in metrics.items():  # sequential: friendlier to free-tier rate limits
        if on_progress:
            on_progress(name)
        results[name] = await _score(metric, sample)
    return results


def evaluate_one(question: str, answer: str, contexts: List[str], reference: str = "",
                 on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    return asyncio.run(_run(question, answer, contexts, (reference or "").strip(), on_progress))
