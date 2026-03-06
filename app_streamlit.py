"""
app_streamlit.py — Swiggy Annual Report RAG Q&A App
=====================================================
Step 2 of the RAG pipeline:
- Loads the FAISS index built by ingest.py
- Accepts user questions via Streamlit UI
- Retrieves relevant chunks (semantic search)
- Generates grounded answers via Groq (Llama 3, free API)
- Displays answer + source pages + supporting context

Run with:
    streamlit run app_streamlit.py
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INDEX_DIR  = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K      = 8   # number of chunks to retrieve

SYSTEM_PROMPT = """You are a precise Q&A assistant for the Swiggy Annual Report FY 2023-24.

Rules:
1. Answer ONLY using the provided CONTEXT below.
2. Do NOT use any outside knowledge.
3. If the answer is not in the context, reply exactly:
   "I could not find this information in the Swiggy Annual Report."
4. Do NOT guess or hallucinate.
5. Cite page numbers when available.
"""

# ─────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Swiggy Annual Report RAG",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .big-title  { font-size: 36px; font-weight: 700; color: #fc5200; }
    .sub-title  { font-size: 15px; color: #888; margin-top: -10px; margin-bottom: 20px; }
    .answer-box {
        padding: 16px 20px;
        border-radius: 8px;
        background-color: #f9f9f9;
        border-left: 5px solid #fc5200;
        font-size: 15px;
        line-height: 1.7;
        color: #222;
    }
    .section-label {
        font-size: 18px;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 6px;
        color: #333;
    }
    .page-badge {
        display: inline-block;
        background: #fff3ec;
        color: #fc5200;
        border: 1px solid #fc5200;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 13px;
        font-weight: 600;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD RESOURCES (cached — loaded once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    """Load FAISS index + embedding model."""
    if not os.path.exists(INDEX_DIR):
        st.error(f"❌ FAISS index not found at '{INDEX_DIR}/'. Run `python ingest.py` first.")
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


@st.cache_resource
def load_llm():
    """Load Groq LLM (Llama 3.1 8B — free tier)."""
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0)


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────
def format_context(docs) -> tuple[str, list[int]]:
    """Combine retrieved docs into a context string and collect page numbers."""
    context = ""
    pages = set()
    for doc in docs:
        page = doc.metadata.get("page")
        if page is not None:
            pages.add(page + 1)   # pypdf is 0-indexed
        context += doc.page_content + "\n\n"
    return context.strip(), sorted(pages)


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="big-title">📊 Swiggy Annual Report RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">FY 2023-24 · Retrieval-Augmented Generation · Groq (Llama 3.1) · FAISS · HuggingFace Embeddings</div>', unsafe_allow_html=True)
st.markdown("---")

# Load env vars (.env file)
load_dotenv()

# Load models
vectorstore = load_vectorstore()
llm         = load_llm()
retriever   = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K}
)

# ─────────────────────────────────────────────
# UI — SIDEBAR (suggested questions)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Try These Questions")
    suggested = [
        "What was Swiggy's total revenue in FY 2023-24?",
        "How many cities does Swiggy food delivery operate in?",
        "What is Instamart and how many dark stores does it have?",
        "Who are the board members of Swiggy?",
        "Was there any fraud discovered at Swiggy?",
        "What was the net loss of Swiggy in FY24?",
        "Did Swiggy declare any dividend?",
        "What are Swiggy's subsidiaries?",
    ]
    for q in suggested:
        if st.button(q, use_container_width=True):
            st.session_state["query_input"] = q
            st.session_state["auto_submit"] = True

    st.markdown("---")
    st.markdown("**Source:** [Swiggy Annual Report FY 2023-24](https://www.swiggy.com/about-us/)")
    st.markdown("**Model:** Llama 3.1 8B (Groq)")
    st.markdown("**Embeddings:** all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# UI — MAIN QUERY INPUT
# ─────────────────────────────────────────────

# Initialize session state keys
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "auto_submit" not in st.session_state:
    st.session_state["auto_submit"] = False

query = st.text_input(
    "🔍 Enter your question about the Swiggy Annual Report:",
    value=st.session_state["query_input"],
    placeholder="e.g. What was Swiggy's net loss in FY 2023-24?",
    key="query_input",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Submit", type="primary", use_container_width=True)

# Trigger automatically when sidebar button was clicked
if st.session_state.get("auto_submit"):
    st.session_state["auto_submit"] = False
    submit = True

# ─────────────────────────────────────────────
# RAG PIPELINE — on submit
# ─────────────────────────────────────────────
active_query = st.session_state.get("query_input", "").strip()

if submit and active_query:
    query = active_query
    start_time = time.time()

    with st.spinner("🔍 Retrieving relevant sections and generating answer..."):

        # STEP 3A: Retrieve top-K chunks via semantic similarity
        docs = retriever.invoke(query)

        if not docs:
            st.markdown('<div class="section-label">Final Answer</div>', unsafe_allow_html=True)
            st.error("I could not find this information in the Swiggy Annual Report.")
        else:
            # STEP 3B: Format context from retrieved chunks
            context, pages = format_context(docs)

            # STEP 3C: Call Groq LLM with context-grounded prompt
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""QUESTION:
{query}

CONTEXT:
{context}

Answer clearly and concisely. Cite page numbers where possible.""")
            ]
            response  = llm.invoke(messages)
            answer    = response.content.strip()

            # ── Final Answer ──────────────────────────
            st.markdown('<div class="section-label">✅ Final Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            # ── Source Pages ──────────────────────────
            st.markdown('<div class="section-label">📄 Source Pages</div>', unsafe_allow_html=True)
            if pages:
                badges = " ".join([f'<span class="page-badge">Page {p}</span>' for p in pages])
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.write("No page numbers identified.")

            # ── Supporting Context (expandable) ───────
            with st.expander("🔍 Supporting Context — Top Retrieved Chunks"):
                for i, doc in enumerate(docs[:4], 1):
                    pg = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Chunk {i} — Page {pg}**")
                    st.write(doc.page_content)
                    st.markdown("---")

    elapsed = round(time.time() - start_time, 2)
    st.caption(f"⏱ Response time: {elapsed}s · Answers strictly from retrieved document context.")

elif submit and not active_query:
    st.warning("Please enter a question first.")
