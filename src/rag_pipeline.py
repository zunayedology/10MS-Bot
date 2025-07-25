import torch
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from config import FAISS_INDEX_DIR, EMBEDDING_MODEL, TOP_K

def setup_rag_pipeline():
    try:
        # Load FAISS
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": TOP_K})

        # Load Qwen Bangla LLM
        model_name = "BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        tokenizer.pad_token = tokenizer.eos_token

        def generate_response(context: str, question: str):
            messages = [
                {"role": "system", "content": "তুমি একটি বাংলা AI সহকারী, শুধুমাত্র প্রদত্ত প্রসঙ্গ ব্যবহার করে উত্তর দাও।"},
                {"role": "user", "content": f"প্রসঙ্গ:\n{context}\n\nপ্রশ্ন:\n{question}\n\nউত্তর:"}
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

            return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        # Define RAG function
        def rag_fn(question: str):
            docs = retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in docs])[:1000]
            return generate_response(context, question)

        return rag_fn

    except Exception as e:
        print(f"Pipeline error: {e}")
        return None