from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = REPO_ROOT / "webapp" / "static"
DEFAULT_CHART = REPO_ROOT / "charts" / "ymca_extra.json"

app = FastAPI(title="JustDanceOpenCV Web UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _load_chart(chart_path: Path) -> dict[str, Any]:
    with chart_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_media_path(chart_data: dict[str, Any]) -> Path | None:
    raw_path = chart_data.get("video_path")
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate

    # fallback: resolve relative to repo if authoring machine absolute path is stale
    repo_candidate = REPO_ROOT / Path(raw_path).name
    if repo_candidate.exists():
        return repo_candidate
    return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/chart")
def chart() -> dict[str, Any]:
    chart_data = _load_chart(DEFAULT_CHART)
    moves = chart_data.get("moves", [])
    media_path = _resolve_media_path(chart_data)

    simplified_moves = [
        {
            "name": str(m.get("name", "")),
            "start_ms": int(m.get("start_ms", 0)),
            "norm_xy": ((m.get("pose") or {}).get("norm_xy") or {}),
        }
        for m in moves
    ]

    return {
        "title": chart_data.get("title", "Untitled"),
        "offset_ms": int(chart_data.get("offset_ms", 0)),
        "move_count": len(simplified_moves),
        "moves": simplified_moves,
        "media_available": bool(media_path),
        "media_url": "/api/media/video" if media_path else None,
    }


@app.get("/api/media/video")
def media_video() -> FileResponse:
    chart_data = _load_chart(DEFAULT_CHART)
    media_path = _resolve_media_path(chart_data)
    if media_path is None:
        raise HTTPException(status_code=404, detail="No playable reference media found for chart video_path")
    return FileResponse(media_path)
