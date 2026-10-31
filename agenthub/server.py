"""FastAPI HTTP layer.  pip install -e ".[server]"  then:  uvicorn agenthub.server:app

POST /v1/chat            {message, session_id?, role?}            -> result (may be status=awaiting_approval)
POST /v1/chat/stream     same body                                 -> Server-Sent Events
POST /v1/approve         {session_id, decisions: {call_id: bool}}  -> result
GET  /health
"""
from __future__ import annotations

import json

from .agent import AgentError
from .bootstrap import build_agent
from .service import AgentService


def create_app(service: AgentService | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    svc = service or AgentService(build_agent())
    app = FastAPI(title="AgentHub", version="0.1.0")

    class ChatIn(BaseModel):
        message: str = Field(min_length=1, max_length=4000)
        session_id: str | None = None
        role: str = "support"

    class ApproveIn(BaseModel):
        session_id: str
        decisions: dict[str, bool]

    @app.get("/health")
    def health():
        return svc.health()

    @app.post("/v1/chat")
    def chat(body: ChatIn):
        try:
            return svc.chat(body.message, body.session_id, body.role)
        except AgentError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/v1/approve")
    def approve(body: ApproveIn):
        try:
            return svc.approve(body.session_id, body.decisions)
        except AgentError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/v1/chat/stream")
    def chat_stream(body: ChatIn):
        def gen():
            try:
                for ev in svc.stream(body.message, body.session_id, body.role):
                    yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except AgentError as e:
                yield f"event: error\ndata: {json.dumps({'detail': str(e)}, ensure_ascii=False)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def __getattr__(name: str):  # lazy `app` so importing this module doesn't require fastapi
    if name == "app":
        return create_app()
    raise AttributeError(name)
