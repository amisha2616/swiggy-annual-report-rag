"""
ingest.py — Document Processing & Vector Store Builder
=======================================================
Improved version with better chunking for financial tables.
Run this ONCE before starting the app:
    python ingest.py
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


ANNUAL_REPORT_PATH = "data/Annual-Report-FY-2023-24.pdf"
INDEX_DIR          = "faiss_index"
EMBED_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE         = 1200   # increased to keep financial tables intact
CHUNK_OVERLAP      = 200    # increased overlap to avoid missing context at boundaries



def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



SECTION_MAP = [
    (range(1, 4),    "Corporate Information"),
    (range(4, 30),   "Board's Report and Financial Summary"),
    (range(30, 43),  "Auditor's Report Standalone"),
    (range(43, 99),  "Standalone Financial Statements"),
    (range(99, 109), "Auditor's Report Consolidated"),
    (range(109, 168),"Consolidated Financial Statements"),
]

def get_section(page_num: int) -> str:
    for page_range, name in SECTION_MAP:
        if page_num in page_range:
            return name
    return "Annual Report"



def main():
    if not os.path.exists(ANNUAL_REPORT_PATH):
        raise FileNotFoundError(f"PDF not found at: {ANNUAL_REPORT_PATH}")

   
    print(f"[1/4] Loading PDF: {ANNUAL_REPORT_PATH}")
    loader = PyPDFLoader(ANNUAL_REPORT_PATH)
    documents = loader.load()
    print(f"[1/4] Loaded {len(documents)} pages.")


    print("[2/4] Cleaning text and adding metadata...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata["source"]  = "Swiggy Annual Report FY 2023-24"
        doc.metadata["section"] = get_section(doc.metadata.get("page", 0) + 1)

   
    documents = [d for d in documents if len(d.page_content.strip()) > 50]
    print(f"[2/4] {len(documents)} pages retained after filtering.")

    
    print("[3/4] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

   
    for chunk in chunks:
        section = chunk.metadata.get("section", "")
        page    = chunk.metadata.get("page", 0) + 1
        chunk.page_content = f"[Section: {section} | Page {page}]\n{chunk.page_content}"

    print(f"[3/4] Created {len(chunks)} chunks.")


    print(f"[4/4] Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[4/4] Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    print(f"[4/4] FAISS index saved to: {INDEX_DIR}/")
    print("\nDone! Run: py -m streamlit run app_streamlit.py")


if __name__ == "__main__":
    main()
