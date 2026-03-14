import os
import uuid
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from marshmallow import Schema, fields
from flask_cors import CORS
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
)
from rag import RagAgent
from agent_placeholder import Agent


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(ROOT_DIR, "generated-scripts")
GEN_FLOW = os.path.join(ROOT_DIR, "generated-flowcharts")
MEDIA_DIR = os.path.join(ROOT_DIR, "media")
FLOWCHART_MEDIA_DIR = os.path.join(ROOT_DIR, "flowchart_media")


app = Flask(__name__)
CORS(app, resources={r"/*": {"origin": "http://localhost:5500"}})

# constants for theapp
upload_folder = os.path.join(os.path.dirname(__file__), "uploads")

# database config
app.secret_key = os.getenv("secret-key")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("pgsql")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String, unique=True, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)


with app.app_context():
    db.create_all()


class userValidation(Schema):
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


def build_rag_agent(session_id: str) -> RagAgent:
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3.5-7B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.7,
    )
    model = ChatHuggingFace(llm=llm)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    return RagAgent(embeddings=embeddings, model=model, session_id=session_id)


@app.route("/status")
def status():
    return "<p>200,SERVER UP AND RUNNING.</p>"


@app.route("/login", methods=["POST"])
def login():
    # conn = psycopg2.connect(os.getenv('pgsql'))
    data = request.get_json()
    if not data:
        return jsonify({"msg": "no input data, try again"}), 401

    if "username" in data and "email" in data:
        username = data["username"]
        email = data["email"]
    else:
        return jsonify({"success": False, "msg": "username and email are required"}), 400

    try:  # tries to validate input, if failed returns 401
        user = User(username=username, email=email)
        find = User.query.filter(User.username == username, User.email == email).first()
        if find:
            return jsonify(
                {
                    "success": False,
                    "msg": "user already exists, Welcome " + username,
                    "username": username,
                    "email": email,
                }
            )
        valSchema = userValidation()
        input = dict(username=username, email=email)
        check = valSchema.load(input)
        print(check)

        try:  # tries to write to the databse, returns a 503 when failed.
            db.session.add(user)
            db.session.commit()
            session["user_id"] = str(user.id)
            return jsonify(
                {
                    "success": True,
                    "msg": "user added successfully",
                    "username": username,
                    "email": email,
                    "id": session["user_id"],
                }
            ), 200
        except Exception as e:
            print(e)
            return jsonify(
                {
                    "success": False,
                    "msg": "Something went wrong, we're fixing it, Hang in there!",
                    "cause": "database-write",
                }
            ), 500
    except Exception as e:
        print(e)
        print(find)
        return jsonify(
            {"success": False, "msg": "Invalid input credentials, try again."}
        ), 401



@app.route("/manim", methods=["POST", "GET"])
def manim_response():
    payload = request.args.to_dict() if request.method == "GET" else (request.get_json(silent=True) or {})
    query = payload.get("query") or payload.get("prompt") or payload.get("message")
    if not query:
        return jsonify({"ok": False, "error": "Missing query/prompt/message"}), 400

    session_id = payload.get("session_id") or session.get("user_id") or str(uuid.uuid4())
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


# replace all this with a chat endpoint that calls
allowed_files = ["pdf", "xlsx", "docx"]
# post req to receive the files (1) for rag with the prompt, and send the response back.
@app.route("/upload", methods=["POST"])
def rag():
    """
    Docstring for rag

    :return: Description
    :rtype: type[that]
    """
    if request.method == "POST":
        if "file" not in request.files:
            return "no file found, try again", 401

        file = request.files["file"]
        filename = file.filename

        if not filename or not allowed_file(filename):
            return "invalid file", 400

        # the file path to storethe pdf locally.
        filepath = os.path.join(upload_folder, filename)

        os.makedirs(upload_folder, exist_ok=True)
        file.save(filepath)

        print("file saved at:", filepath)

        session_id = request.form.get("session_id") or session.get("user_id") or str(uuid.uuid4())
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
    payload = request.get_json(silent=True) or {}
    query = payload.get("query") or payload.get("prompt") or payload.get("message")
    if not query:
        return jsonify({"ok": False, "error": "Missing query/prompt/message"}), 400

    session_id = payload.get("session_id") or session.get("user_id") or str(uuid.uuid4())
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


@app.route(
    "/response", methods=["POST", "GET"]
)  # unified endpoint for text + video generation
def response():
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

    session_id = payload.get("session_id") or session.get("user_id") or str(uuid.uuid4())
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
