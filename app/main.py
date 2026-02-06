from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.schemas import ChatRequest, ChatResponse
from app.chatbot import Chatbot

app = FastAPI(title="Oromia Gov AI Bot", version="0.1.0")
chatbot = Chatbot()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "service": "Oromia Gov AI Bot"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = chatbot.search(
            query=request.query,
            sector=request.sector,
            language=request.language
        )
        return ChatResponse(
            query=request.query,
            rewritten_query=result["rewritten_query"],
            answer=result["answer"],
            sector=result["sector"],
            language=result["language"],
            confidence=result["confidence"],
            source_file=result.get("source_file")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
