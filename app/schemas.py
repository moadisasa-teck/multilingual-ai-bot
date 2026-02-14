from pydantic import BaseModel, Field
from typing import Optional, List

class ChatTurn(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    query: str = Field(..., example="How do I get a passport?")
    sector: Optional[str] = None
    language: Optional[str] = None
    history: Optional[List[ChatTurn]] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    query: str
    rewritten_query: str
    answer: str
    sector: str
    language: str
    confidence: float
    source_file: Optional[str] = None
    conversation_id: Optional[str] = None
