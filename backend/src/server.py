import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from marshmallow import Schema, fields
from supabase import create_client, Client


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(ROOT_DIR, "generated-scripts")
GEN_FLOW = os.path.join(ROOT_DIR, "generated-flowcharts")
MEDIA_DIR = os.path.join(ROOT_DIR, "media")
FLOWCHART_MEDIA_DIR = os.path.join(ROOT_DIR, "flowchart_media")

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

USE_LOCAL_STORAGE = os.getenv("USE_LOCAL_STORAGE", "false").strip().lower() in {"1", "true", "yes"}
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:8080")

app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

upload_folder = os.path.join(os.path.dirname(__file__), "uploads")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


class UserValidation(Schema):
    username = fields.Str(required=True)
    email = fields.Email(required=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_files


def coerce_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "video", "manim"}:
        return True
    if normalized in {"0", "false", "no", "n", "text"}:
        return False
    return None


def build_rag_agent(session_id: str):
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    from hf_embeddings import HFInferenceEmbeddings
    from rag import RagAgent

    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
    )
    model = ChatHuggingFace(llm=llm)
    embeddings = HFInferenceEmbeddings()
    return RagAgent(embeddings=embeddings, model=model, session_id=session_id)


def _get_agent(session_id: str, create_video, task_id):
    from agent_placeholder import Agent

    return Agent(session_id=session_id, create_video=create_video, task_id=task_id)


@app.route("/media/<path:filename>")
def serve_media(filename):
    return send_from_directory(MEDIA_DIR, filename)


@app.route("/flowchart_media/<path:filename>")
def serve_flowchart(filename):
    return send_from_directory(FLOWCHART_MEDIA_DIR, filename)


@app.route("/generated-scripts/<path:filename>")
def serve_scripts(filename):
    return send_from_directory(GEN_DIR, filename)


@app.route("/generated-flowcharts/<path:filename>")
def serve_gen_flow(filename):
    return send_from_directory(GEN_FLOW, filename)


@app.route("/status")
def status():
    return jsonify({"status": "ok", "service": "glyph-backend"})


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"success": False, "msg": "No input data, try again"}), 400

    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip()

    if not username or not email:
        return jsonify({"success": False, "msg": "username and email are required"}), 400

    try:
        val_schema = UserValidation()
        val_schema.load({"username": username, "email": email})
    except Exception:
        return jsonify({"success": False, "msg": "Invalid input credentials, try again."}), 401

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
            return jsonify({
                "success": True,
                "msg": f"Welcome back, {user['username']}",
                "username": user["username"],
                "email": user["email"],
                "id": user["id"],
            }), 200

        new_id = str(uuid.uuid4())
        supabase.table("users").insert({
            "id": new_id,
            "username": username,
            "email": email,
        }).execute()

        return jsonify({
            "success": True,
            "msg": "User added successfully",
            "username": username,
            "email": email,
            "id": new_id,
        }), 201
    except Exception as e:
        print(f"Supabase login error: {e}")
        return jsonify({
            "success": False,
            "msg": "Something went wrong",
            "cause": "database-write",
        }), 500


@app.route("/manim", methods=["POST", "GET"])
def manim_response():
    from agent_placeholder import Agent

    payload = request.args.to_dict() if request.method == "GET" else (request.get_json(silent=True) or {})
    query = payload.get("query") or payload.get("prompt") or payload.get("message")
    if not query:
        return jsonify({"ok": False, "error": "Missing query/prompt/message"}), 400

    session_id = payload.get("session_id") or str(uuid.uuid4())
    task_id = payload.get("task_id") or payload.get("task_uuid")

    agent = Agent(session_id=session_id, create_video=True, task_id=task_id)
    result = agent.run_agent(query)
    status_code = 200 if result.get("ok") else 500
    return (
        jsonify(
            {
                "ok": bool(result.get("ok")),
                "session_id": session_id,
                "task_id": result.get("task_id", agent.task_id),
                "route": result.get("route", "manim_only"),
                "create_video": True,
                "response": result.get("string"),
                "content": result.get("content"),
                "research": result.get("research"),
                "error": result.get("error"),
                "warnings": result.get("warnings", []),
            }
        ),
        status_code,
    )


allowed_files = ["pdf", "xlsx", "docx"]


@app.route("/upload", methods=["POST"])
def rag():
    if request.method == "POST":
        if "file" not in request.files:
            return "no file found, try again", 401

        file = request.files["file"]
        filename = file.filename

        if not filename or not allowed_file(filename):
            return "invalid file", 400

        filepath = os.path.join(upload_folder, filename)

        os.makedirs(upload_folder, exist_ok=True)
        file.save(filepath)

        print("file saved at:", filepath)

        session_id = request.form.get("session_id") or str(uuid.uuid4())
        rag_agent = build_rag_agent(session_id=session_id)
        stored_doc_ids = rag_agent.store(filepath, session_id=session_id)

        return (
            jsonify(
                {
                    "ok": True,
                    "msg": "upload successful",
                    "session_id": session_id,
                    "filename": filename,
                    "stored_count": len(stored_doc_ids) if stored_doc_ids else 0,
                }
            ),
            200,
        )


@app.route("/agent", methods=["POST"])
def agent_video_only():
    from agent_placeholder import Agent

    payload = request.get_json(silent=True) or {}
    query = payload.get("query") or payload.get("prompt") or payload.get("message")
    if not query:
        return jsonify({"ok": False, "error": "Missing query/prompt/message"}), 400

    session_id = payload.get("session_id") or str(uuid.uuid4())
    task_id = payload.get("task_id") or payload.get("task_uuid")

    agent = Agent(session_id=session_id, create_video=True, task_id=task_id)
    result = agent.run_agent(query)
    status_code = 200 if result.get("ok") else 500

    return (
        jsonify(
            {
                "ok": bool(result.get("ok")),
                "session_id": session_id,
                "task_id": result.get("task_id", agent.task_id),
                "route": result.get("route", "manim_only"),
                "create_video": True,
                "response": result.get("string"),
                "content": result.get("content"),
                "research": result.get("research"),
                "error": result.get("error"),
                "warnings": result.get("warnings", []),
            }
        ),
        status_code,
    )


@app.route("/response", methods=["POST", "GET"])
def response():
    from agent_placeholder import Agent

    payload = request.args.to_dict() if request.method == "GET" else (request.get_json(silent=True) or {})

    query = payload.get("query") or payload.get("prompt") or payload.get("message")
    if not query:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Missing query. Provide one of: query, prompt, message",
                }
            ),
            400,
        )

    session_id = payload.get("session_id") or str(uuid.uuid4())
    requested_mode = payload.get("mode")
    create_video = coerce_optional_bool(payload.get("create_video"))
    if create_video is None and requested_mode:
        mode = str(requested_mode).strip().lower()
        if mode in {"video", "manim"}:
            create_video = True
        elif mode == "text":
            create_video = False

    task_id = payload.get("task_id") or payload.get("task_uuid")

    agent = Agent(
        session_id=session_id,
        create_video=create_video,
        task_id=task_id,
    )
    result = agent.run_agent(query)

    status_code = 200 if result.get("ok") else 500
    resolved_route = result.get("route")
    if not resolved_route:
        if create_video is True:
            resolved_route = "manim_only"
        elif create_video is False:
            resolved_route = "text"
        else:
            resolved_route = "auto"

    response_payload = {
        "ok": bool(result.get("ok")),
        "session_id": session_id,
        "task_id": result.get("task_id", agent.task_id),
        "route": resolved_route,
        "create_video": create_video,
        "response": result.get("string"),
        "content": result.get("content"),
        "research": result.get("research"),
        "error": result.get("error"),
        "warnings": result.get("warnings", []),
    }
    return jsonify(response_payload), status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080", debug=True)
