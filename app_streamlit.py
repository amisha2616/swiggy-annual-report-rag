import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# ----------------------------
# Configuration
# ----------------------------

INDEX_DIR = "faiss_index"

SYSTEM_PROMPT = """
You are an AI assistant answering questions strictly 
from the Swiggy Annual Report.

Rules:
1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not explicitly present in the context, reply:
   "I could not find this information in the Swiggy Annual Report."
4. Do NOT guess.
"""

# ----------------------------
# Styling
# ----------------------------

st.set_page_config(page_title="Swiggy Annual Report RAG", layout="wide")

st.markdown("""
<style>
.big-title {font-size:38px; font-weight:700;}
.answer-box {
    padding: 15px;
    border-radius: 8px;
    background-color: #f4f6f8;
    border-left: 6px solid #2e7d32;
}
.section-title {font-size:22px; font-weight:600; margin-top:20px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helper Functions
# ----------------------------

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def format_context(docs):
    context = ""
    pages = set()

    for doc in docs:
        page = doc.metadata.get("page")
        if page is not None:
            pages.add(page + 1)
        context += doc.page_content + "\n\n"

    return context, sorted(pages)


# ----------------------------
# UI
# ----------------------------

st.markdown('<div class="big-title">📊 Swiggy Annual Report RAG System</div>', unsafe_allow_html=True)
st.write("Ask questions about the Swiggy Annual Report (FY24).")
st.write("---")

load_dotenv()

vectorstore = load_vectorstore()
llm = load_llm()

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

query = st.text_input("Enter your question:")

if st.button("Submit") and query:

    start_time = time.time()

    with st.spinner("Retrieving and generating answer..."):

        docs = retriever.invoke(query)

        if not docs:
            st.markdown('<div class="section-title">Final Answer</div>', unsafe_allow_html=True)
            st.error("I could not find this information in the Swiggy Annual Report.")
        else:
            context, pages = format_context(docs)

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""
QUESTION:
{query}

CONTEXT:
{context}

Answer clearly in 3–5 lines.
""")
            ]

            response = llm.invoke(messages)
            answer_text = response.content.strip()

            # ----------------------------
            # Display Answer
            # ----------------------------

            st.markdown('<div class="section-title">Final Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-box">{answer_text}</div>', unsafe_allow_html=True)

            # ----------------------------
            # Source Pages
            # ----------------------------

            st.markdown('<div class="section-title">Source Pages</div>', unsafe_allow_html=True)
            if pages:
                st.write(", ".join([f"Page {p}" for p in pages]))
            else:
                st.write("No source pages identified.")

            # ----------------------------
            # Supporting Context
            # ----------------------------

            with st.expander("🔍 Supporting Context (Top Retrieved Chunks)"):
                for i, doc in enumerate(docs[:3], 1):
                    st.markdown(f"**Chunk {i} — Page {doc.metadata.get('page', 0)+1}**")
                    st.write(doc.page_content)
                    st.markdown("---")

    end_time = time.time()
    st.caption(f"Response time: {round(end_time - start_time, 2)} seconds")
    st.caption("Answers are generated strictly from retrieved document context.")