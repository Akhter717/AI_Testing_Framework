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

import numpy as np
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

_IDK = ("don't know", "do not know", "dont know", "cannot find", "can't find",
        "not mentioned", "not provided", "no information", "not in the context",
        "not contain", "insufficient")


def is_idk(text: str) -> bool:
    """True if an answer is a refusal such as "I don't know"."""
    t = (text or "").lower()
    return len(t) < 220 and any(p in t for p in _IDK)


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


async def _run(question, answer, contexts, reference, on_progress, answer_metrics):
    judge, emb = _build()
    sample = SingleTurnSample(
        user_input=question, response=answer,
        retrieved_contexts=contexts, reference=reference or None,
    )
    # Groq does not support n>1 completions -> strictness=1
    metrics = {}
    if answer_metrics:
        metrics["Faithfulness"] = Faithfulness(llm=judge)
        metrics["Answer relevancy"] = ResponseRelevancy(llm=judge, embeddings=emb, strictness=1)
        metrics["Context precision (no reference)"] = LLMContextPrecisionWithoutReference(llm=judge)
    if reference:
        metrics["Context precision (with reference)"] = LLMContextPrecisionWithReference(llm=judge)
        metrics["Context recall"] = LLMContextRecall(llm=judge)
        if answer_metrics:
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
                 on_progress: Optional[Callable[[str], None]] = None,
                 answer_metrics: bool = True) -> Dict[str, object]:
    """answer_metrics=False -> only the reference-based RETRIEVAL metrics (used when the RAG said "I don't know")."""
    return asyncio.run(_run(question, answer, contexts, (reference or "").strip(), on_progress, answer_metrics))


# =====================================================================
# EXPLANATIONS: re-compute every metric step by step so the user can SEE why a score came out as it did.
# These are transparent re-computations (judge LLM verdicts + simple formulas); they can differ a little
# from the official RAGAS number because both come from separate LLM runs.
# =====================================================================
def _ask_json(prompt: str):
    out = rag.chat(rag.judge_model(), "You output only valid JSON, with no other text.", prompt)
    m = re.search(r"(\{.*\}|\[.*\])", out, re.S)
    if not m:
        raise ValueError("Judge did not return JSON.")
    return json.loads(m.group(1))


def _numbered(chunks: List[rag.Chunk]) -> str:
    return "\n\n".join(f"[Chunk {i}] ({c.source} p.{c.page})\n{c.text}" for i, c in enumerate(chunks, 1))


def _band(v: float) -> str:
    return "high" if v >= 0.8 else ("medium" if v >= 0.5 else "low")


def _cos(a: str, b: str) -> float:
    v = rag.embed_texts([a, b])
    return float(max(0.0, min(1.0, np.dot(v[0], v[1]))))


def explain_faithfulness(claims: List[Dict]) -> Dict:
    n = len(claims)
    sup = sum(1 for c in claims if c["verdict"] == "supported")
    score = sup / n if n else 0.0
    rows = [{"Claim in the RAG answer": c["claim"],
             "Supported by chunks?": "yes" if c["verdict"] == "supported" else "NO",
             "Evidence from chunks": c["evidence"]} for c in claims]
    bad = [c["claim"] for c in claims if c["verdict"] != "supported"]
    why = f"{sup} of {n} claims in the answer are supported by the retrieved chunks, so {sup}/{n} = {score:.3f} ({_band(score)})."
    if bad:
        why += " Not supported (possible hallucination): " + "; ".join(f'"{b}"' for b in bad) + "."
    elif n:
        why += " Every claim can be traced to the chunks, so there is no hallucination."
    return {"value": score, "rows": rows, "formula": "Faithfulness = supported claims / total claims", "why": why}


def explain_answer_relevancy(question: str, answer: str) -> Dict:
    implied = rag.chat(rag.judge_model(), "You write questions.",
                       f"Here is an answer:\n{answer}\n\nWrite the single question that this answer is a direct "
                       "response to. Output only the question.").strip().strip('"')
    score = _cos(question, implied)
    rows = [{"Your question": question, "Question the answer actually responds to": implied,
             "Closeness in meaning (cosine similarity)": f"{score:.3f}"}]
    why = (f"The judge read only the answer and guessed which question it answers. That guessed question is "
           f"{_band(score)}ly close in meaning to your real question (similarity {score:.3f}). ")
    why += ("A high value means the answer stays on topic." if score >= 0.8 else
            "A lower value means the answer drifted, was too vague or answered something else.")
    return {"value": score, "rows": rows,
            "formula": "Answer relevancy = cosine similarity(your question, question implied by the answer)", "why": why}


def explain_context_precision(question: str, target: str, chunks: List[rag.Chunk], target_label: str) -> Dict:
    data = _ask_json(
        f"Question: {question}\nTarget answer: {target}\n\nChunks in retrieval order:\n{_numbered(chunks)}\n\n"
        f"For EACH of the {len(chunks)} chunks decide whether it contains information useful for producing the "
        'target answer. Reply with a JSON array: [{"chunk": 1, "useful": true, "reason": "short reason"}, ...]')
    flags = {int(d["chunk"]): bool(d.get("useful")) for d in data if isinstance(d, dict) and "chunk" in d}
    reasons = {int(d["chunk"]): str(d.get("reason", "")) for d in data if isinstance(d, dict) and "chunk" in d}
    hits, precs, rows = 0, [], []
    for i, c in enumerate(chunks, 1):
        u = flags.get(i, False)
        if u:
            hits += 1
            precs.append(hits / i)
        rows.append({"Rank": i, "Chunk": f"{c.source} p.{c.page}", "Useful?": "yes" if u else "no",
                     "Precision at this rank": f"{hits}/{i} = {hits / i:.2f}" if u else "-",
                     "Reason": reasons.get(i, "")})
    score = sum(precs) / len(precs) if precs else 0.0
    if precs:
        why = (f"{len(precs)} of {len(chunks)} chunks were useful. Average of the precision at each useful rank "
               f"({', '.join(f'{p:.2f}' for p in precs)}) = {score:.3f} ({_band(score)}). "
               "Useful chunks ranked higher raise the score; useless chunks ranked first lower it.")
    else:
        why = "None of the retrieved chunks was useful for the target answer, so the score is 0."
    return {"value": score, "rows": rows,
            "formula": f"Context precision = average of (useful chunks so far / rank) over the useful ranks. Judged against: {target_label}.",
            "why": why}


def explain_context_recall(reference: str, chunks: List[rag.Chunk]) -> Dict:
    data = _ask_json(
        f"Reference answer:\n{reference}\n\nRetrieved chunks:\n{_numbered(chunks)}\n\n"
        "Split the reference answer into short factual statements. For each statement say whether it can be found "
        'in the chunks. Reply with a JSON array: [{"statement": "...", "found": true, "evidence": "short quote or empty"}, ...]')
    rows, found = [], 0
    for d in data:
        if not isinstance(d, dict):
            continue
        f = bool(d.get("found"))
        found += f
        rows.append({"Statement from the reference answer": str(d.get("statement", "")),
                     "Found in retrieved chunks?": "yes" if f else "NO", "Evidence": str(d.get("evidence", ""))})
    n = len(rows)
    score = found / n if n else 0.0
    missing = [r["Statement from the reference answer"] for r in rows if r["Found in retrieved chunks?"] == "NO"]
    why = f"{found} of {n} statements of the reference answer are present in the retrieved chunks, so {found}/{n} = {score:.3f} ({_band(score)})."
    if missing:
        why += " Missing from the chunks: " + "; ".join(f'"{m}"' for m in missing) + ". Retrieval did not bring this information."
    return {"value": score, "rows": rows, "formula": "Context recall = reference statements found in the chunks / total reference statements", "why": why}


def explain_factual_correctness(answer: str, reference: str) -> Dict:
    d = _ask_json(
        f"RAG answer:\n{answer}\n\nReference answer:\n{reference}\n\n"
        "Split BOTH texts into short factual claims. For each claim of the RAG answer say whether the reference "
        "answer supports it. For each claim of the reference answer say whether the RAG answer states it. Reply with JSON: "
        '{"answer_claims": [{"claim": "...", "in_reference": true}], "reference_claims": [{"claim": "...", "in_answer": true}]}')
    ac = [x for x in d.get("answer_claims", []) if isinstance(x, dict)]
    rc = [x for x in d.get("reference_claims", []) if isinstance(x, dict)]
    tp_p = sum(1 for x in ac if x.get("in_reference"))
    tp_r = sum(1 for x in rc if x.get("in_answer"))
    prec = tp_p / len(ac) if ac else 0.0
    rec = tp_r / len(rc) if rc else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rows = [{"Source": "RAG answer claim", "Claim": str(x.get("claim", "")),
             "Matches the other answer?": "yes" if x.get("in_reference") else "NO"} for x in ac]
    rows += [{"Source": "Reference answer claim", "Claim": str(x.get("claim", "")),
              "Matches the other answer?": "yes" if x.get("in_answer") else "NO"} for x in rc]
    why = (f"Precision = {tp_p}/{len(ac)} = {prec:.2f} (how much of the RAG answer is backed by the reference). "
           f"Recall = {tp_r}/{len(rc)} = {rec:.2f} (how much of the reference the RAG answer covers). "
           f"F1 = {f1:.3f} ({_band(f1)}).")
    return {"value": f1, "rows": rows, "formula": "Factual correctness = F1 of claim precision and claim recall", "why": why}


def explain_semantic_similarity(answer: str, reference: str) -> Dict:
    score = _cos(answer, reference)
    rows = [{"RAG answer": answer, "Reference answer": reference, "Cosine similarity of meaning": f"{score:.3f}"}]
    why = f"The two answers are {_band(score)}ly similar in meaning (cosine similarity {score:.3f}). This ignores exact wording and only compares meaning."
    return {"value": score, "rows": rows, "formula": "Semantic similarity = cosine similarity of the two answers' embeddings", "why": why}


def explain_all(question: str, answer: str, chunks: List[rag.Chunk], reference: str,
                names: List[str], claims: Optional[List[Dict]] = None,
                on_progress: Optional[Callable[[str], None]] = None) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for name in names:
        if on_progress:
            on_progress(name)
        try:
            if name == "Faithfulness":
                out[name] = explain_faithfulness(claims if claims else check_claims(answer, chunks))
            elif name == "Answer relevancy":
                out[name] = explain_answer_relevancy(question, answer)
            elif name == "Context precision (no reference)":
                out[name] = explain_context_precision(question, answer, chunks, "the RAG answer")
            elif name == "Context precision (with reference)":
                out[name] = explain_context_precision(question, reference, chunks, "the reference answer")
            elif name == "Context recall":
                out[name] = explain_context_recall(reference, chunks)
            elif name == "Factual correctness":
                out[name] = explain_factual_correctness(answer, reference)
            elif name == "Semantic similarity":
                out[name] = explain_semantic_similarity(answer, reference)
        except Exception as ex:
            out[name] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
    return out
