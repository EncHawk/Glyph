import os
import subprocess
import sys
import jwt
import boto3
import botocore
import uuid
import datetime
from flask import Flask, request, render_template, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID
from marshmallow import Schema, fields
from flask_cors import CORS
import manim_agent as Manim
from pydantic import BaseModel, constr
import manim
from rag import RagAgent
from AGENT import Agent


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


@app.route("/status")
def home():
    return "<p>200,SERVER UP AND RUNNING.</p>"


@app.route("/login", methods=["POST"])
def login():
    # conn = psycopg2.connect(os.getenv('pgsql'))
    data = request.get_json()
    if not data:
        return jsonify({"msg": "no input data, try again"}), 401

    if "username" and "email" in data:
        username = data["username"]
        email = data["email"]

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


@app.route("/agent", methods=["POST"]) # meant exclusively for manim / future diffusion inclusion
def agent():
    input = request.get_json()
    query = input.get("prompt")
    if not query:
        return jsonify({"msg": "No input found, try sending smtn next time"}), 403
    # this prompt must go in the agent's class.
    user_id = session.get("user_id")
    try:
        # TODO may be this can be taken as input to add as many as the user wants.
        agent = Agent(session_id=user_id, attempts=3)
        agent_response = agent.run_agent(query=query)  # has data, and tool payload
        if not agent_response:
            return {"msg": "agent returned null", "data": agent_response}, 500

        # instead of making the aws upload from teh agent class, we do it in separate functions instead.
        # aws_string = agent_response.string
        if isinstance(agent_response, tuple):
            agent_response = agent_response[0]
        print(agent_response)
        return agent_response, 200
    except Exception as e:
        print("=" * 50)
        print("EXCEPTION IN CHAT ENDPOINT:")
        print("Exception type:", type(e).__name__)
        print("Exception message:", repr(str(e)))
        print("Exception args:", e.args)
        import traceback

        traceback.print_exc()
        print("=" * 50)
        error = str(e)
        if not error:
            error = f"{type(e).__name__} (no message)"
        return jsonify(
            {"msg": "something went wrong, we're working on it!", "cause": error}
        ), 500


@app.route("/manim", methods=["POST", "GET"])
def manimResponse():
    class ModelResponse(BaseModel):
        code: constr(min_length=1, strip_whitespace=True)
        className: str

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Model_Response",
            "schema": ModelResponse.model_json_schema(),
            "strict": True,
        },
    }
    try:
        # data = request.get_json()
        # input = data.get('prompt')
        # print(input)
        # print("api gateway")
        # manimInstance = Manim(response_format=response_format, prompt = input)
        # res = manimInstance.inferModel(model=ModelResponse)
        # base_dir = os.path.dirname(os.path.abspath(__file__))
        # gen_dir = os.path.join(base_dir, "generated-scripts")
        # os.makedirs(gen_dir, exist_ok=True)
        # gen = os.path.join(gen_dir, f"{res.className}.py")

        # if len(res.code) > 10:
        #     print('writing to a file')
        #     with open (gen, 'w') as f:
        #         f.write(res.code)

        # file = os.path.abspath(f"generated-scripts/{res.className}.py")
        # WE ARE GONNA USE THE AGENT HERE INSTEAD OF AN ACTUAL CALL, WE USE THE AGENT TO DO THE WORK AND RETURN AN AWS STRING

        try:
            subprocess.run(["manim", "-pql", file, res.className], check=True)
            return {
                "success": "true",
                "msg": "looking good",
                "data": f"{request.remote_addr}",
                "file": f"{res.className}.py",
            }, 200

        except Exception as e:
            return {
                "success": "false",
                "msg": "not looking good, couldnt run command",
                "data": f"{request.remote_addr}",
            }, 503

        # os.system(f"manim -pql {file} {res.className}")
    except Exception as e:
        print(e)
        return {
            "success": "false",
            "msg": f"something went wrong while generating, try again.",
            "data": f"{request.remote_addr}",
        }, 503


# replace all this with a chat endpoint that calls
allowed_files = ["pdf", "xlsx", "docx"]


@app.route(
    "/upload", methods=["POST"]
)  # post req to receive the files (1) for rag with the prompt, and send the response back.
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

        file.save(filepath)
        rag = RagAgent()

        print("file saved at:", filepath)

        rag.store(filepath)
        rag.infer()

        return "upload successful", 200
    else:
        return "GET the fuck out, invalid request method.", 403


@app.route("/response", methods=["GET"])
def rag_response():
    """
    returns the response that is generated from RAG class
    todo : add special methods in rag class that generates the response,
    then return that response form this endpoint
    """
    llm = HuggingFaceEndpoint(  # add your huggingface token, this shit free and good heck yeah!
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.7,
    )
    model = ChatHuggingFace(llm=llm)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    rag = RagAgent(embeddings=embeddings, model=model)
    rag.store()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080", debug=True)
