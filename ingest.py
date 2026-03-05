import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

ANNUAL_REPORT_PATH = "data/Annual-Report-FY-2023-24 (1) (1).pdf"
INDEX_DIR = "faiss_index"


import re

def clean_text(text: str) -> str:
    # Keep line breaks (tables need them), just remove excessive spaces per line
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)   # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)  # limit huge gaps
    return text.strip()

def main():
    if not os.path.exists(ANNUAL_REPORT_PATH):
        raise FileNotFoundError(f"PDF not found at: {ANNUAL_REPORT_PATH}")

    print("Loading PDF...")
    loader = PyPDFLoader(ANNUAL_REPORT_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    # 🔹 Preprocess text
    print("Cleaning text...")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    # 🔹 Meaningful chunking
    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    # 🔹 Add metadata enrichment (optional but good practice)
    for chunk in chunks:
        chunk.metadata["source"] = "Swiggy Annual Report FY24"

    # 🔹 Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 🔹 Build FAISS index
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"Saving index to: {INDEX_DIR}")
    vectorstore.save_local(INDEX_DIR)

    print("Document processing completed successfully ✅")


if __name__ == "__main__":
    main()