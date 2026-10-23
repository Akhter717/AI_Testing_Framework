import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="RAG + RAGAS Evaluator", layout="wide")

# ---- secrets: Streamlit Cloud -> st.secrets ; local -> .env ----
KEY_SOURCE = ".env / environment"
try:
    for _k in ("GROQ_API_KEY", "RAG_MODEL", "JUDGE_MODEL", "EMBED_MODEL"):
        if _k in st.secrets:
            if not os.getenv(_k):
                os.environ[_k] = str(st.secrets[_k]).strip()
            if _k == "GROQ_API_KEY":
                KEY_SOURCE = "Streamlit secrets"
except Exception:
    pass

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. On Streamlit Cloud open the app -> Settings -> Secrets and add:\n\nGROQ_API_KEY = \"gsk_...\"")
    st.stop()

import rag
import evaluator

METRIC_INFO = {
    "Faithfulness": ("Is every claim in the RAG answer supported by the retrieved chunks? (hallucination check)", "No"),
    "Answer relevancy": ("Does the RAG answer actually address the question?", "No"),
    "Context precision (no reference)": ("Were the retrieved chunks useful for answering, and ranked well?", "No"),
    "Context precision (with reference)": ("Same, judged against the reference answer", "Yes"),
    "Context recall": ("Do the retrieved chunks contain everything the reference answer needs?", "Yes"),
    "Factual correctness": ("How well do the RAG answer's claims match the reference answer's claims?", "Yes"),
    "Semantic similarity": ("How close in meaning are the RAG answer and the reference answer?", "Yes"),
}


def badge(v):
    if not isinstance(v, float):
        return "⚠️"
    return "🟢" if v >= 0.8 else ("🟡" if v >= 0.5 else "🔴")


# ================================================================ header
st.title("RAG + RAGAS Evaluator")
h1, h2, h3 = st.columns(3)
h1.info("**① APP**\n\nYou upload a PDF and ask a question (typed or random).")
h2.success(f"**② RAG SYSTEM**\n\nChunk → embed → vector store → retrieve → answer\n\n"
           f"Answer LLM: `{rag.rag_model()}`")
h3.warning(f"**③ RAGAS EVALUATION**\n\nSeparate judge LLM writes a reference answer, compares, scores\n\n"
           f"Judge LLM: `{rag.judge_model()}`")

# ================================================================ sidebar (APP inputs)
with st.sidebar:
    st.header("① APP - Inputs")
    st.caption(f"Groq key: loaded from {KEY_SOURCE}")
    files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    st.subheader("Settings")
    top_k = st.slider("Chunks given to RAG LLM (top-k)", 1, 10, 4)
    wide_k = st.slider("Chunks given to judge for reference answer", 4, 20, 10,
                       help="The judge sees more of the document than the RAG LLM, so its answer is a fairer reference.")
    run_ragas = st.checkbox("Run RAGAS metrics", value=True)
    run_claims = st.checkbox("Run claim-by-claim hallucination check", value=True)

# ================================================================ build knowledge base automatically
if files:
    sig = tuple((f.name, f.size) for f in files)
    if st.session_state.get("kb", {}).get("sig") != sig:
        with st.spinner("② RAG SYSTEM: chunking, embedding and building the vector store..."):
            try:
                chunks = []
                for f in files:
                    chunks += rag.chunk_pdf(f.getvalue(), f.name)
                idx = rag.RagIndex(chunks)
                st.session_state.kb = {"sig": sig, "index": idx, "files": [f.name for f in files]}
                st.session_state.pop("result", None)
            except Exception as e:
                st.session_state.pop("kb", None)
                st.error(f"Could not process the PDF: {e}")
else:
    st.session_state.pop("kb", None)

kb = st.session_state.get("kb")

# ================================================================ ① APP: ask
with st.container(border=True):
    st.subheader("① APP - Ask a question")
    if not kb:
        st.info("Upload a PDF in the sidebar to start.")
    else:
        def _random_q():
            try:
                st.session_state["question_input"] = evaluator.make_question(kb["index"].chunks)
                st.session_state.pop("q_error", None)
            except Exception as e:
                st.session_state["q_error"] = str(e)

        st.text_input("Question (type your own, or generate one from the PDF)", key="question_input")
        if st.session_state.get("q_error"):
            st.error(f"Could not generate a question: {st.session_state['q_error']}")
        st.text_area("Your own reference answer (optional)", key="user_reference", height=80,
                     help="Leave empty and the judge LLM will write a reference answer from the document.")
        b1, b2, _ = st.columns([1, 1, 3])
        b1.button("🎲 Random question from PDF", on_click=_random_q)
        ask = b2.button("Ask & Evaluate", type="primary",
                        disabled=not st.session_state.get("question_input", "").strip())

# ================================================================ run the pipeline
if kb and ask:
    question = st.session_state["question_input"].strip()
    user_ref = st.session_state.get("user_reference", "").strip()
    res = {"question": question, "rag_model": rag.rag_model(), "judge_model": rag.judge_model()}
    with st.status("Running...", expanded=True) as status:
        try:
            st.write("② Retrieving chunks from the vector store...")
            pairs = kb["index"].retrieve(question, top_k)
            ctx = [c for c, _ in pairs]
            res["pairs"] = pairs
            st.write("② RAG LLM is writing the answer...")
            res["answer"] = rag.answer(question, ctx)

            if user_ref:
                res["reference"], res["ref_source"] = user_ref, "you"
            else:
                st.write("③ Judge LLM is writing its own reference answer from a wider context...")
                wide = [c for c, _ in kb["index"].retrieve(question, wide_k)]
                res["reference"] = evaluator.generate_reference(question, wide)
                res["ref_source"] = "judge"
            ref_ok = evaluator.NO_ANSWER not in res["reference"]
            ref_for_metrics = res["reference"] if ref_ok else ""

            res["claims"], res["claims_error"] = None, None
            if run_claims:
                st.write("③ Checking the answer claim by claim...")
                try:
                    res["claims"] = evaluator.check_claims(res["answer"], ctx)
                except Exception as e:
                    res["claims_error"] = str(e)

            res["scores"] = None
            if run_ragas:
                scores = evaluator.evaluate_one(
                    question, res["answer"], [c.text for c in ctx], ref_for_metrics,
                    on_progress=lambda n: st.write(f"③ RAGAS: {n}..."),
                )
                res["scores"] = scores
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            res = None
    if res:
        st.session_state.result = res

res = st.session_state.get("result")

# ================================================================ ② RAG SYSTEM
with st.container(border=True):
    st.subheader("② RAG SYSTEM")
    if not kb:
        st.caption("Waiting for a PDF.")
    else:
        idx = kb["index"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PDF files", len(kb["files"]))
        m2.metric("Chunks created", len(idx.chunks))
        m3.metric("Embedding dimensions", idx.dim)
        m4.metric("Vector store", "FAISS" if idx.index is not None else "NumPy")
        st.caption(f"Embedding model: `{rag.embed_model_name()}` | Store: {idx.store_name} | "
                   f"Chunk size {rag.CHUNK_SIZE} chars, overlap {rag.CHUNK_OVERLAP}")
        with st.expander("Preview first 3 chunks"):
            for c in idx.chunks[:3]:
                st.markdown(f"**{c.source} - page {c.page}**")
                st.write(c.text)
        if res:
            st.markdown(f"**Question:** {res['question']}")
            st.markdown(f"**Retrieved chunks (top-{len(res['pairs'])}) - check these against the answer:**")
            for i, (c, s) in enumerate(res["pairs"], 1):
                with st.expander(f"{i}. {c.source} - page {c.page}  |  similarity {s:.3f}"):
                    st.write(c.text)
            st.markdown("**RAG answer**")
            st.write(res["answer"])
            st.caption(f"Generated by `{res['rag_model']}` using {len(res['pairs'])} retrieved chunks")

# ================================================================ ③ RAGAS EVALUATION
with st.container(border=True):
    st.subheader("③ RAGAS EVALUATION")
    if not res:
        st.caption("Results appear here after you click **Ask & Evaluate**.")
    else:
        ref = res["reference"]
        ref_ok = evaluator.NO_ANSWER not in ref
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**RAG answer** (system under test)")
            st.write(res["answer"])
            st.caption(f"`{res['rag_model']}`")
        with c2:
            who = "your reference answer" if res["ref_source"] == "you" else "judge LLM's own answer"
            st.markdown(f"**Reference answer** ({who})")
            st.write(ref if ref_ok else "The judge could not find the answer in the document.")
            st.caption("Written by you" if res["ref_source"] == "you"
                       else f"Written by `{res['judge_model']}` from a wider context - not human verified")

        # ---- verdict
        dont_know = "don't know" in res["answer"].lower() or "do not know" in res["answer"].lower()
        if dont_know:
            st.warning("The RAG system answered \"I don't know\". Either the PDF has no answer, or retrieval missed "
                       "it - raise top-k or check the retrieved chunks.")
        claims = res.get("claims")
        if claims:
            bad = [c for c in claims if c["verdict"] != "supported"]
            if bad:
                st.error(f"Possible hallucination: {len(bad)} of {len(claims)} claims are NOT supported by the retrieved chunks.")
            else:
                st.success(f"All {len(claims)} claims in the answer are supported by the retrieved chunks.")
        elif res.get("claims_error"):
            st.warning(f"Claim check failed: {res['claims_error']}")

        if claims:
            with st.expander("Claim-by-claim check", expanded=True):
                st.table([{"Claim": c["claim"],
                           "Verdict": "✅ supported" if c["verdict"] == "supported" else "❌ not supported",
                           "Evidence from chunks": c["evidence"]} for c in claims])

        scores = res.get("scores")
        if scores:
            st.markdown("**RAGAS metrics** (0 to 1, higher is better)")
            rows = []
            for name, val in scores.items():
                info = METRIC_INFO.get(name, ("", ""))
                rows.append({"": badge(val), "Metric": name,
                             "Score": round(val, 3) if isinstance(val, float) else val,
                             "What it tells you": info[0], "Needs reference": info[1]})
            st.table(rows)
            if not ref_ok:
                st.caption("Reference-based metrics were skipped because no reference answer could be produced.")
            st.caption("Scores come from an LLM judge. They vary slightly between runs - use them as signals, not proof. "
                       "Checks are made against the retrieved chunks, not the whole PDF.")
            csv = "metric,score\n" + "\n".join(f'"{k}",{v}' for k, v in scores.items())
            st.download_button("Download scores (CSV)", csv, "ragas_scores.csv", "text/csv")
