import streamlit as st
import os
from pathlib import Path
from extract_text import process_pdfs
from process_data import load_and_chunk_texts
from vectorize import vectorize_and_store
from rag_pipeline import setup_rag_pipeline

# Set project root directory (parent of src/)
PROJECT_ROOT = str(Path(__file__).parent.parent)
PDF_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
TEXT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FAISS_INDEX_DIR = os.path.join(PROJECT_ROOT, "faiss_index")

# Create data/raw if it doesn't exist
os.makedirs(PDF_DIR, exist_ok=True)

# Function to check if pipeline needs to run
def should_run_pipeline():
    # Check if PDFs exist
    pdf_exists = any(f.endswith(".pdf") for f in os.listdir(PDF_DIR))
    # Check if processed texts exist
    text_exists = os.path.exists(TEXT_DIR) and len(os.listdir(TEXT_DIR)) > 0
    # Check if FAISS index exists
    faiss_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    return pdf_exists and not (text_exists and faiss_exists)

# Cache the RAG pipeline
@st.cache_resource
def initialize_pipeline():
    if should_run_pipeline():
        with st.spinner("Processing PDFs..."):
            process_pdfs()
        with st.spinner("Chunking texts..."):
            load_and_chunk_texts()
        with st.spinner("Vectorizing texts..."):
            vectorize_and_store()
    with st.spinner("Setting up RAG pipeline..."):
        return setup_rag_pipeline()

# Set up Streamlit page
st.title("10 Minute School Bangla AI")

# Check if PDFs exist
if not any(f.endswith(".pdf") for f in os.listdir(PDF_DIR)):
    st.error(f"No PDFs found in {PDF_DIR}. Please add textbook PDFs to {PDF_DIR} and restart.")
    st.stop()

# Initialize RAG pipeline
if "rag_chain" not in st.session_state:
    try:
        st.session_state.rag_chain = initialize_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new questions
if prompt := st.chat_input("Enter your question:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from RAG pipeline
    with st.spinner("Thinking..."):
        try:
            result = st.session_state.rag_chain.invoke({"question": prompt})
            response = result["answer"]
        except Exception as e:
            response = f"Error: {str(e)}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)