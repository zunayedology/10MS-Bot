from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import setup_rag_pipeline
import uuid

app = FastAPI()
rag_chain = setup_rag_pipeline()

class Query(BaseModel):
    question: str
    session_id: str = str(uuid.uuid4())  # Default unique session ID

@app.post("/ask")
async def ask_question(query: Query):
    try:
        result = rag_chain.invoke(
            {"question": query.question},
            config={"configurable": {"session_id": query.session_id}}
        )
        return {"answer": result["answer"]}
    except Exception as e:
        return {"error": str(e)}