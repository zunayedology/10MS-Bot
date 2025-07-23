import streamlit as st
from rag_pipeline import setup_rag_pipeline

# Set up Streamlit page
st.title("Bangla Textbook RAG Chatbot")
st.write("Ask questions about Bangla textbooks (7th to 12th grade) in English or Bengali.")

# Initialize RAG pipeline
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = setup_rag_pipeline()

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
        result = st.session_state.rag_chain({"question": prompt})
        response = result["answer"]

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)