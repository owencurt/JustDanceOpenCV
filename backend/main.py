import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.session import GameSession

app = FastAPI(title="JustDance Browser Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHARTS_DIR = Path("charts")
DEFAULT_SCORING = str(CHARTS_DIR / "ymca.json")
DEFAULT_CHOREO = str(CHARTS_DIR / "ymca_extra.json")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/charts")
def list_charts():
    charts = sorted([p.name for p in CHARTS_DIR.glob("*.json")])
    return {
        "charts": charts,
        "default_scoring": Path(DEFAULT_SCORING).name,
        "default_choreo": Path(DEFAULT_CHOREO).name,
    }


@app.websocket("/ws/game")
async def game_socket(websocket: WebSocket):
    await websocket.accept()
    session = None
    try:
        init_raw = await websocket.receive_text()
        init_msg = json.loads(init_raw)
        scoring = init_msg.get("scoring_chart", Path(DEFAULT_SCORING).name)
        choreo = init_msg.get("choreo_chart", Path(DEFAULT_CHOREO).name)

        scoring_path = str(CHARTS_DIR / scoring)
        choreo_path = str(CHARTS_DIR / choreo)
        session = GameSession(scoring_chart=scoring_path, choreo_chart=choreo_path)

        await websocket.send_json({"type": "ready", "state": session.state})

        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "command":
                session.command(msg.get("action", ""))
                await websocket.send_json({"type": "ack", "action": msg.get("action"), "state": session.state})
                continue

            if msg_type == "frame":
                payload = session.process_frame(msg.get("image_b64", ""))
                await websocket.send_json({"type": "frame_result", **payload})
                continue

            await websocket.send_json({"type": "error", "error": "unknown_message_type"})

    except WebSocketDisconnect:
        pass
    finally:
        if session:
            session.close()


@app.exception_handler(Exception)
async def catch_all(_, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})
