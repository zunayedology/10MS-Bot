from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline,HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
import torch
from config import MODEL_NAME, EMBEDDING_MODEL, FAISS_INDEX_DIR, TOP_K

def setup_rag_pipeline():
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    model = model.to(f"cuda:{device}" if device >= 0 else "cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True, device=device)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Load FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    # Set up conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create RAG chain
    crc = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return crc

def test_rag_pipeline(crc):
    test_cases = [
        {"question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "expected": "শুম্ভুনাথ"},
        {"question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "expected": "মামাকে"},
        {"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "expected": "১৫ বছর"}
    ]
    for test in test_cases:
        result = crc({"question": test["question"]})
        print(f"Question: {test['question']}")
        print(f"Expected: {test['expected']}")
        print(f"Got: {result['answer']}")

if __name__ == "__main__":
    rag_chain = setup_rag_pipeline()
    test_rag_pipeline(rag_chain)