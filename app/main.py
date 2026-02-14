from datetime import datetime, UTC
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.chatbot import Chatbot
from app.schemas import ChatRequest, ChatResponse, ChatTurn

app = FastAPI(title="Oromia Gov AI Bot", version="0.1.0")
chatbot = Chatbot()

SESSION_COOKIE = "chat_session_id"
MAX_HISTORY_MESSAGES = 50
MAX_CONVERSATIONS = 50
MAX_TITLE_LENGTH = 72

SESSION_STORE: dict[str, dict] = {}
SESSION_LOCK = Lock()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


@app.get("/settings")
def settings_page():
    return FileResponse("static/settings.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "Oromia Gov AI Bot"}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _get_or_create_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get(SESSION_COOKIE)
    if session_id:
        return session_id

    session_id = str(uuid4())
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 7,
    )
    return session_id


def _new_conversation_id() -> str:
    return str(uuid4())


def _trim_history(items: list[dict]) -> list[dict]:
    return items[-MAX_HISTORY_MESSAGES:]


def _sanitize_title(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "New Chat"
    if len(cleaned) > MAX_TITLE_LENGTH:
        return f"{cleaned[:MAX_TITLE_LENGTH - 1]}…"
    return cleaned


def _get_or_create_session_state(session_id: str) -> dict:
    state = SESSION_STORE.get(session_id)
    if state:
        return state

    state = {"active_id": None, "conversations": []}
    SESSION_STORE[session_id] = state
    return state


def _find_conversation(state: dict, conversation_id: str) -> dict | None:
    for conv in state["conversations"]:
        if conv["id"] == conversation_id:
            return conv
    return None


def _conversation_summary(conv: dict) -> dict:
    return {
        "id": conv["id"],
        "title": conv["title"],
        "created_at": conv["created_at"],
        "updated_at": conv["updated_at"],
        "message_count": len(conv["messages"]),
    }


def _create_conversation(state: dict, title: str = "New Chat") -> dict:
    now = _now_iso()
    conv = {
        "id": _new_conversation_id(),
        "title": _sanitize_title(title),
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    state["conversations"].insert(0, conv)
    state["active_id"] = conv["id"]
    state["conversations"] = state["conversations"][:MAX_CONVERSATIONS]
    return conv


@app.get("/history")
def get_history(
    request: Request,
    response: Response,
    conversation_id: str | None = Query(default=None),
):
    session_id = _get_or_create_session_id(request, response)
    with SESSION_LOCK:
        state = _get_or_create_session_state(session_id)
        target_id = conversation_id or state.get("active_id")
        target = _find_conversation(state, target_id) if target_id else None

        if target_id and not target:
            target_id = None
            target = None

        conversations = [_conversation_summary(conv) for conv in state["conversations"]]
        return {
            "active_id": target_id,
            "conversations": conversations,
            "messages": target["messages"] if target else [],
        }


@app.post("/history/new")
def create_history(request: Request, response: Response):
    session_id = _get_or_create_session_id(request, response)
    with SESSION_LOCK:
        state = _get_or_create_session_state(session_id)
        conversation = _create_conversation(state)
        return {
            "conversation": _conversation_summary(conversation),
            "active_id": conversation["id"],
            "messages": [],
        }


@app.delete("/history")
def clear_history(
    request: Request,
    response: Response,
    conversation_id: str | None = Query(default=None),
):
    session_id = _get_or_create_session_id(request, response)
    with SESSION_LOCK:
        state = _get_or_create_session_state(session_id)
        if conversation_id:
            state["conversations"] = [c for c in state["conversations"] if c["id"] != conversation_id]
            if state.get("active_id") == conversation_id:
                state["active_id"] = state["conversations"][0]["id"] if state["conversations"] else None
            return {"cleared": True, "active_id": state.get("active_id")}

        state["conversations"] = []
        state["active_id"] = None
        return {"cleared": True, "active_id": None}


@app.post("/chat", response_model=ChatResponse)
def chat(request_data: ChatRequest, request: Request, response: Response):
    try:
        session_id = _get_or_create_session_id(request, response)
        with SESSION_LOCK:
            state = _get_or_create_session_state(session_id)
            conversation_id = request_data.conversation_id or state.get("active_id")
            conversation = _find_conversation(state, conversation_id) if conversation_id else None

            if not conversation:
                conversation = _create_conversation(state)
                conversation_id = conversation["id"]
            state["active_id"] = conversation_id

            session_history = conversation["messages"].copy()

        incoming_history = [item.dict() for item in request_data.history] if request_data.history else []
        combined_history = session_history if session_history else incoming_history
        combined_history = _trim_history(combined_history)

        result = chatbot.search(
            query=request_data.query,
            sector=request_data.sector,
            language=request_data.language,
            history=combined_history,
        )

        user_turn = ChatTurn(role="user", content=request_data.query, timestamp=_now_iso()).dict()
        assistant_turn = ChatTurn(role="assistant", content=result["answer"], timestamp=_now_iso()).dict()
        updated_history = _trim_history(combined_history + [user_turn, assistant_turn])

        with SESSION_LOCK:
            state = _get_or_create_session_state(session_id)
            conv = _find_conversation(state, conversation_id)
            if not conv:
                conv = _create_conversation(state)
                conversation_id = conv["id"]

            if conv["title"] in {"New Chat", ""} and not conv["messages"]:
                conv["title"] = _sanitize_title(request_data.query)
            conv["messages"] = updated_history
            conv["updated_at"] = _now_iso()

            state["conversations"].sort(key=lambda c: c["updated_at"], reverse=True)
            state["active_id"] = conversation_id

        return ChatResponse(
            query=request_data.query,
            rewritten_query=result["rewritten_query"],
            answer=result["answer"],
            sector=result["sector"],
            language=result["language"],
            confidence=result["confidence"],
            source_file=result.get("source_file"),
            conversation_id=conversation_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
