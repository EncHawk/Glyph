import os
import sys
import uuid
from pathlib import Path

import modal
from flask import Flask
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Modal app definition (merged from app.py)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "libcairo2-dev",
        "libpango1.0-dev",
        "pkg-config",
        "poppler-utils",
        "tesseract-ocr",
        "shared-mime-info",
    )
    .pip_install_from_pyproject(str(ROOT_DIR / "pyproject.toml"))
    .env({"PYTHONPATH": f"{str(ROOT_DIR)}:{str(ROOT_DIR / 'src')}"})
)

volume = modal.Volume.from_name("glyph-data", create_if_missing=True)
VOLUME_PATH = "/glyph-data"

secrets = [
    modal.Secret.from_dotenv(str(ROOT_DIR / ".env")),
]

app = modal.App("glyph-backend", image=image, secrets=secrets)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

GEN_DIR = str(ROOT_DIR / "generated-scripts")
GEN_FLOW = str(ROOT_DIR / "generated-flowcharts")
MEDIA_DIR = str(ROOT_DIR / "media")
UPLOAD_DIR = str(ROOT_DIR / "uploads")


def _get_session_id(payload: dict) -> str:
    return payload.get("session_id") or str(uuid.uuid4())


def _resolve_query(payload: dict) -> str | None:
    return payload.get("query") or payload.get("prompt") or payload.get("message")

# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

@app.function(volumes={VOLUME_PATH: volume})
def status():
    return {"status": "ok", "service": "glyph-backend"}


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
)
def login(username: str, email: str):
    import uuid as _uuid

    from marshmallow import Schema, fields
    from supabase import create_client

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)

    class UserValidation(Schema):
        username = fields.Str(required=True)
        email = fields.Email(required=True)

    try:
        schema = UserValidation()
        schema.load({"username": username, "email": email})
    except Exception:
        return {"success": False, "msg": "Invalid input credentials, try again."}

    try:
        existing = (
            supabase.table("users")
            .select("id, username, email")
            .or_(f"username.eq.{username},email.eq.{email}")
            .limit(1)
            .execute()
        )
        if existing.data:
            user = existing.data[0]
            return {
                "success": True,
                "msg": f"Welcome back, {user['username']}",
                "username": user["username"],
                "email": user["email"],
                "id": user["id"],
            }

        new_id = str(_uuid.uuid4())
        supabase.table("users").insert({
            "id": new_id,
            "username": username,
            "email": email,
        }).execute()

        return {
            "success": True,
            "msg": "User added successfully",
            "username": username,
            "email": email,
            "id": new_id,
        }
    except Exception:
        return {
            "success": False,
            "msg": "Something went wrong",
            "cause": "database-write",
        }


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
)
def agent_run(
    query: str,
    session_id: str | None = None,
    task_id: str | None = None,
    create_video: bool | None = None,
):
    from src.agent_placeholder import Agent

    agent = Agent(
        session_id=session_id or str(uuid.uuid4()),
        create_video=create_video,
        task_id=task_id,
    )
    return agent.run_agent(query)


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
)
def upload_file_stream(filename: str, file_bytes: bytes, session_id: str | None = None):
    from src.server import build_rag_agent

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filepath = os.path.join(UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    sid = session_id or str(uuid.uuid4())
    rag_agent = build_rag_agent(session_id=sid)
    stored = rag_agent.store(filepath, session_id=sid)

    return {
        "ok": True,
        "msg": "upload successful",
        "session_id": sid,
        "filename": filename,
        "stored_count": len(stored) if stored else 0,
    }


# ---------------------------------------------------------------------------
# Web endpoint (wraps the existing Flask app)
# ---------------------------------------------------------------------------


def build_flask_app() -> Flask:
    from src.server import app as flask_app

    CORS(flask_app, resources={r"/*": {"origin": "*"}})
    return flask_app


@app.function(
    volumes={VOLUME_PATH: volume},
    secrets=secrets,
)
@modal.asgi_app()
def web():
    return build_flask_app()


# ---------------------------------------------------------------------------
# Local dev entry point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    from src.server import app as flask_app

    flask_app.run(host="0.0.0.0", port=8080, debug=True)
