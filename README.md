# 📊 Swiggy Annual Report RAG System
🔎 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) based Question Answering system built on the Swiggy Annual Report (FY 2023–24).

The system allows users to ask natural language questions and receive accurate, context-grounded answers strictly derived from the document. It prevents hallucination by ensuring responses are generated only from retrieved document content.

# 🚀 Features
`📄 PDF Document Processing

✂️ Intelligent text chunking

🧠 Semantic embeddings using SentenceTransformers

🔍 Vector similarity search using FAISS

🤖 LLM-powered answer generation via Groq API

🛑 Strict context-based answering (no external knowledge)

📑 Source page attribution

📊 Semantic similarity score display

💻 Interactive Streamlit Web UI

⏱ Response time tracking`



# 🏗 System Architecture


`graph TD
    A[User Query] --> B[FAISS Semantic Retrieval]
    B --> C[Top-K Relevant Chunks]
    C --> D[Context Injection into Groq LLM]
    D --> E[Grounded Answer Generation]
    E --> F[Answer + Source Pages + Context]`


# 🛠 Tech Stack


`
Python

Streamlit

LangChain

FAISS (Vector Store)

SentenceTransformers (all-MiniLM-L6-v2)

Groq API (llama-3.1-8b-instant) 

`


⚠ Note:
.env, data/, and faiss_index/ are excluded for security and size reasons.

# ⚙️ Installation & Setup
1️⃣ Clone the Repository
`git clone https://github.com/amisha2616/swiggy-annual-report-rag.git`

`cd swiggy-annual-report-rag`

2️⃣ Create Virtual Environment

`python -m venv .venv`

`.venv\Scripts\activate`

3️⃣ Install Dependencies
`pip install -r requirements.txt`

4️⃣ Add Groq API Key

Create a .env file in the project root: `GROQ_API_KEY=your_groq_api_key_here`

5️⃣ Build Vector Index

Place the Swiggy Annual Report PDF inside a data/ folder and run:

`python ingest.py`

6️⃣ Run the Application
`streamlit run app_streamlit.py`

Then open:

`http://localhost:8501`
## 🧠 How It Works
1️⃣ Document Processing

Loads the Swiggy Annual Report PDF.

Cleans and splits it into meaningful chunks.

Stores metadata including page numbers.

2️⃣ Embedding & Vector Storage

Generates semantic embeddings using SentenceTransformers.

Stores embeddings in FAISS for fast similarity search.

3️⃣ Retrieval-Augmented Generation (RAG)

Retrieves top-k semantically similar chunks.

Injects retrieved context into Groq LLM.

Enforces strict context-based answering.

Displays supporting context and source pages.

4️⃣ Question Answering Interface

Interactive Streamlit UI.

Displays:

Final Answer

Source Pages

Supporting Context

Semantic Similarity Score

Response Time


## 👩‍💻 Author

Developed as part of an AI/ML internship assignment.

