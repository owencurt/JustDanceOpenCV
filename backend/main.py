import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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
DEFAULT_MEDIA = "charts/video_path.mp4"


def _to_web_media_url(raw_path: str | None) -> str | None:
    if not raw_path:
        return None

    raw = raw_path.strip()
    if not raw:
        return None

    if raw.startswith(("http://", "https://")):
        return raw

    normalized = raw.replace("\\", "/")

    if normalized.startswith("/"):
        return normalized

    if normalized.startswith("charts/"):
        return "/" + quote(normalized)

    charts_idx = normalized.find("/charts/")
    if charts_idx != -1:
        return quote(normalized[charts_idx:])

    return "/" + quote(normalized)


def _chart_meta(chart_path: Path) -> dict:
    with chart_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    media_path = data.get("media_path") or data.get("video_path")
    media_url = _to_web_media_url(media_path)
    if not media_url and chart_path.name == Path(DEFAULT_CHOREO).name:
        media_url = _to_web_media_url(DEFAULT_MEDIA)

    return {
        "title": data.get("title") or chart_path.stem,
        "schema_version": data.get("schema_version"),
        "media_path": media_path,
        "media_url": media_url,
    }


if CHARTS_DIR.exists():
    app.mount("/charts", StaticFiles(directory=CHARTS_DIR), name="charts")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/charts")
def list_charts():
    chart_paths = sorted(CHARTS_DIR.glob("*.json"))
    charts = [p.name for p in chart_paths]
    chart_meta = {p.name: _chart_meta(p) for p in chart_paths}
    return {
        "charts": charts,
        "chart_meta": chart_meta,
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
