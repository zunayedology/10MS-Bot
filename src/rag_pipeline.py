from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline,HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import torch
from config import MODEL_NAME, EMBEDDING_MODEL, FAISS_INDEX_DIR, TOP_K

def setup_rag_pipeline():
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    if device == -1:
        print("Warning: GPU not detected, using CPU")
    model = model.to(f"cuda:{device}" if device >= 0 else "cpu")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        device=device,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # Load FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    # Set up conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Custom prompt template for precise answers
    prompt_template = """
    Use the following context to answer the question in Bengali. Provide a concise and accurate answer based on the context. If the answer is not in the context, say "I don't know."

    Context: {context}

    Question: {question}

    Answer: 
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    # Create RAG chain
    crc = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return crc

def test_rag_pipeline(crc):
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শুম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"}
    ]
    for test in test_cases:
        result = crc.invoke({"question": test["question"]})
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected']}")
        print(f"Got: {result['answer']}")

if __name__ == "__main__":
    rag_chain = setup_rag_pipeline()
    test_rag_pipeline(rag_chain)