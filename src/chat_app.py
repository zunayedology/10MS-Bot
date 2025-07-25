import streamlit as st
import os
from extract_text import process_pdfs
from process_data import load_and_chunk_texts
from vectorize import vectorize_and_store
from rag_pipeline import setup_rag_pipeline
from config import PDF_DIR, TEXT_DIR, FAISS_INDEX_DIR

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Helper: Check pipeline needs
def should_run_pipeline():
    pdf_exists = any(f.endswith(".pdf") for f in os.listdir(PDF_DIR))
    faiss_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    return pdf_exists and not faiss_exists

# Initialize RAG pipeline
@st.cache_resource
def initialize_pipeline():
    if should_run_pipeline():
        with st.spinner("Extracting text from PDFs..."):
            process_pdfs()
        with st.spinner("Chunking text..."):
            load_and_chunk_texts()
        with st.spinner("Vectorizing text..."):
            vectorize_and_store()
    with st.spinner("Loading Bangla Qwen LLM pipeline..."):
        rag_fn = setup_rag_pipeline()
        if rag_fn is None:
            raise RuntimeError("Failed to initialize RAG pipeline.")
        return rag_fn

st.title("10 Minute School Bangla AI")
st.write("Ask questions about the Bangla textbook (English or Bangla).")

# Check PDFs
if not any(f.endswith(".pdf") for f in os.listdir(PDF_DIR)):
    st.warning(f"No PDFs found in {PDF_DIR}. Please add at least one PDF.")
    st.stop()

# Initialize pipeline
if "rag_fn" not in st.session_state:
    try:
        st.session_state.rag_fn = initialize_pipeline()
    except Exception as e:
        st.error(f"Pipeline initialization failed: {e}")
        st.stop()

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Enter your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        try:
            response = st.session_state.rag_fn(prompt)
        except Exception as e:
            response = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
