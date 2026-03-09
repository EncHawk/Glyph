import datetime
import json
import operator
import os
import re
import subprocess
import uuid
from typing import List, Optional
from urllib.parse import unquote, urlparse
import boto3
import botocore
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, constr
from typing_extensions import Annotated, TypedDict

from manim_agent import Manim
from rag import RagAgent


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(ROOT_DIR, "generated-scripts")
MEDIA_DIR = os.path.join(ROOT_DIR, "media")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "glyph-data-storage")


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)
model = ChatHuggingFace(llm=llm)


def _slugify(value: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug[:max_len] or "scene").strip("-")


def _safe_class_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_")
    if not cleaned:
        cleaned = "GeneratedScene"
    if not re.match(r"[A-Za-z_]", cleaned):
        cleaned = f"Scene_{cleaned}"
    return cleaned[:80]


def _build_task_id(task_id: Optional[str] = None) -> str:
    if task_id:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", task_id).strip("-")
        return sanitized[:80] or task_id
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _task_date_parts(task_id: str) -> tuple[str, str, str]:
    timestamp = task_id.split("-")[0]
    if len(timestamp) >= 8 and timestamp[:8].isdigit():
        return timestamp[:4], timestamp[4:6], timestamp[6:8]
    now = datetime.datetime.now(datetime.UTC)
    return now.strftime("%Y"), now.strftime("%m"), now.strftime("%d")


def _build_scene_s3_key(
    task_id: str, scene_index: int, scene_id: str, class_name: str
) -> str:
    year, month, day = _task_date_parts(task_id)
    scene_slug = _slugify(scene_id)
    class_slug = _slugify(class_name)
    file_name = f"scene-{scene_index:02d}-{scene_slug}-{class_slug}.mp4"
    return f"manim/{year}/{month}/{day}/{task_id}/{file_name}"


def _build_final_s3_key(task_id: str, scene_count: int) -> str:
    year, month, day = _task_date_parts(task_id)
    return f"final/{year}/{month}/{day}/{task_id}/final-{scene_count:02d}-scenes.mp4"


def _extract_s3_key_from_url(url: str, bucket_name: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path.lstrip("/"))

    if parsed.netloc.startswith(f"{bucket_name}.s3"):
        return path

    bucket_prefix = f"{bucket_name}/"
    if path.startswith(bucket_prefix):
        return path[len(bucket_prefix) :]

    marker = f"{bucket_name}.s3.amazonaws.com/"
    if marker in url:
        return unquote(url.split(marker, 1)[1])

    raise ValueError(f"Could not parse S3 key from url: {url}")


def _render_manim(script_path: str, class_name: str) -> str:
    os.makedirs(MEDIA_DIR, exist_ok=True)
    subprocess.run(
        ["manim", "-ql", "--media_dir", MEDIA_DIR, script_path, class_name],
        check=True,
    )
    local_video = os.path.join(
        MEDIA_DIR,
        "videos",
        os.path.splitext(os.path.basename(script_path))[0],
        "480p15",
        f"{class_name}.mp4",
    )
    if not os.path.exists(local_video):
        raise FileNotFoundError(f"Rendered video not found at {local_video}")
    return local_video


def _upload_to_s3(local_path: str, s3_key: str, bucket_name: str = S3_BUCKET_NAME) -> str:
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("aws_region"),
    )
    try:
        s3.Bucket(bucket_name).upload_file(local_path, s3_key)
    except botocore.exceptions.ClientError as exc:
        raise RuntimeError(f"S3 upload failed for key {s3_key}") from exc
    return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"


@tool
def manim_tool(
    prompt: str,
    class_name_hint: Optional[str] = None,
    task_id: Optional[str] = None,
    scene_id: Optional[str] = None,
    scene_index: int = 0,
    correction_context: Optional[dict] = None,
):
    """
    Generate Manim code, render once, and upload one scene video to S3.
    """

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

    run_task_id = _build_task_id(task_id)

    try:
        final_prompt = prompt
        if class_name_hint:
            final_prompt = f"""{final_prompt}

IMPORTANT:
- Use exactly this class name for the Manim scene: {class_name_hint}
- Return exactly one Scene class.
"""
        if correction_context:
            final_prompt = f"""
{final_prompt}

IMPORTANT: Previous attempt failed with this error:
{correction_context.get("error", "Unknown error")}

Previous code that failed:
```python
{correction_context.get("code", "")}
```

Fix the error and regenerate valid Manim code for the same scene.
"""

        instance = Manim(response_format=response_format, prompt=final_prompt)
        response_text = instance.inferModel(model=ModelResponse)

        target_class = _safe_class_name(class_name_hint or response_text.className)
        generated_code = response_text.code
        generated_code, replaced = re.subn(
            r"class\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*Scene[^)]*)\)\s*:",
            f"class {target_class}(\\1):",
            generated_code,
            count=1,
        )
        if not replaced:
            target_class = _safe_class_name(response_text.className)

        script_dir = os.path.join(GEN_DIR, run_task_id)
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, f"{target_class}.py")
        with open(script_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(generated_code)

        local_video = _render_manim(script_path, target_class)

        resolved_scene_id = scene_id or f"scene_{scene_index:02d}"
        s3_key = _build_scene_s3_key(
            task_id=run_task_id,
            scene_index=max(scene_index, 1),
            scene_id=resolved_scene_id,
            class_name=target_class,
        )
        aws_string = _upload_to_s3(local_video, s3_key, S3_BUCKET_NAME)
        return {
            "ok": True,
            "msg": "Scene rendered and uploaded",
            "string": aws_string,
            "code": generated_code,
            "className": target_class,
            "task_id": run_task_id,
            "s3_key": s3_key,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "code": response_text.code if "response_text" in locals() else None,
            "className": (
                class_name_hint
                if class_name_hint
                else (response_text.className if "response_text" in locals() else None)
            ),
            "task_id": run_task_id,
        }


@tool
def contextual_text_tool(prompt: str, session_id: str):
    """
    Generate a text response using RAG (store + retrieve + answer).
    """
    rag_llm = HuggingFaceEndpoint(
        repo_id="zai-org/GLM-4.7",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.3,
    )
    rag_model = ChatHuggingFace(llm=rag_llm)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    rag_agent = RagAgent(embeddings=embeddings, model=rag_model, session_id=session_id)

    # Store incoming query as session memory, then retrieve and answer.
    rag_agent.store(source=prompt, session_id=session_id, is_text=True)
    return rag_agent.infer(query=prompt, session_id=session_id)


class Scene(TypedDict):
    id: str
    type: str
    class_name: str
    prompt: str
    s3_url: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    scene_plan: List[Scene]
    current_index: int
    completed_urls: Annotated[list[str], operator.add]
    final_video_url: Optional[str]
    route: Optional[str]
    task_id: str
    session_id: str


tools = [manim_tool]
tools_by_name = {tool_obj.name: tool_obj for tool_obj in tools}
model_with_tools = ChatHuggingFace(llm=llm).bind_tools(tools=tools)


def classifier_node(state: AgentState) -> AgentState:
    """Classifies user intent and sets route unless route is pre-selected."""
    if state.get("route"):
        return {"llm_calls": state.get("llm_calls", 0)}

    user_msg = state["messages"][-1].content
    prompt = f"""You are a router. Given the user message, output ONLY one of:
- manim_only
- text

User: {user_msg}
Route:"""
    route = model.invoke([SystemMessage(content=prompt)]).content.strip().lower()
    if route not in ("manim_only", "text"):
        route = "text"
    return {"route": route, "llm_calls": state.get("llm_calls", 0) + 1}


def planning_node(state: AgentState) -> AgentState:
    """Creates a scene plan for video generation."""
    user_msg = state["messages"][-1].content
    prompt = f"""You are a video scene planner for an educational explainer video.
Given the topic below, output ONLY a valid JSON array of scenes.

Rules:
- Each scene has keys: id, type, class_name, prompt
- type must be "manim"
- 3 to 5 scenes
- Start with introduction and end with wrap-up
- No markdown, no explanation, JSON only

Topic: {user_msg}
"""
    response = model.invoke([SystemMessage(content=prompt)]).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()

    try:
        scene_plan = json.loads(response)
    except json.JSONDecodeError:
        scene_plan = [
            {
                "id": "scene_01",
                "type": "manim",
                "class_name": "IntroductionScene",
                "prompt": user_msg,
            },
            {
                "id": "scene_02",
                "type": "manim",
                "class_name": "WrapUpScene",
                "prompt": f"Summarize and conclude: {user_msg}",
            },
        ]

    if not isinstance(scene_plan, list) or len(scene_plan) == 0:
        scene_plan = [
            {
                "id": "scene_01",
                "type": "manim",
                "class_name": "IntroductionScene",
                "prompt": user_msg,
            }
        ]

    task_id = state["task_id"]
    task_suffix = task_id.split("-")[-1][:8]
    normalized_plan: List[Scene] = []

    for index, raw_scene in enumerate(scene_plan, start=1):
        scene_id = raw_scene.get("id", f"scene_{index:02d}")
        class_name_base = _safe_class_name(
            raw_scene.get("class_name", f"Scene{index:02d}")
        )
        class_name = f"{class_name_base}_{task_suffix}_{index:02d}"
        scene_prompt = raw_scene.get("prompt", user_msg)

        normalized_plan.append(
            {
                "id": scene_id,
                "type": "manim",
                "class_name": class_name,
                "prompt": scene_prompt,
                "s3_url": None,
            }
        )

    return {
        "scene_plan": normalized_plan,
        "current_index": 0,
        "completed_urls": [],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def scene_executor_node(state: AgentState) -> AgentState:
    """Renders one scene and uploads it to S3."""
    index = state["current_index"]
    scene = state["scene_plan"][index]

    result = manim_tool.invoke(
        {
            "prompt": scene["prompt"] + "Do not make any errors, ensure that the errors are minimal",
            "class_name_hint": scene["class_name"],
            "task_id": state["task_id"],
            "scene_id": scene["id"],
            "scene_index": index + 1,
            "correction_context": None,
        }
    )

    if not result.get("ok"):
        raise RuntimeError(
            f"Scene generation failed for {scene['id']}: {result.get('error', 'Unknown error')}"
        )

    s3_url = result["string"]
    updated_plan = list(state["scene_plan"])
    updated_plan[index] = {**scene, "s3_url": s3_url}

    return {
        "scene_plan": updated_plan,
        "current_index": index + 1,
        "completed_urls": [s3_url],
    }


def ffmpeg_node(state: AgentState) -> AgentState:
    """Downloads scene videos, stitches them, uploads final video."""
    urls = state["completed_urls"]
    if not urls:
        raise RuntimeError("No scene videos available for stitching")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("aws_region"),
    )

    task_id = state["task_id"]
    local_files = []
    for index, url in enumerate(urls, start=1):
        s3_key = _extract_s3_key_from_url(url, S3_BUCKET_NAME)
        local_path = f"/tmp/{task_id}_scene_{index:02d}.mp4"
        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        local_files.append(local_path)

    concat_path = f"/tmp/{task_id}_concat_list.txt"
    with open(concat_path, "w", encoding="utf-8") as file_obj:
        for file_path in local_files:
            file_obj.write(f"file '{file_path}'\n")

    output_path = f"/tmp/{task_id}_final.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-c",
                "copy",
                output_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                output_path,
            ],
            check=True,
        )

    final_key = _build_final_s3_key(task_id, len(local_files))
    final_url = _upload_to_s3(output_path, final_key, S3_BUCKET_NAME)

    return {
        "final_video_url": final_url,
        "messages": [HumanMessage(content=f"Final video ready: {final_url}")],
    }


def text_node(state: AgentState) -> AgentState:
    """Handles pure text responses."""
    result = contextual_text_tool.invoke(
        {
            "prompt": state["messages"][-1].content,
            "session_id": state["session_id"],
        }
    )
    return {"messages": [HumanMessage(content=str(result))]}


def route_after_classify(state: AgentState) -> str:
    return state.get("route", "text")


def route_after_scene(state: AgentState) -> str:
    if state["current_index"] < len(state["scene_plan"]):
        return "scene_executor_node"
    return "ffmpeg_node"


builder = StateGraph(AgentState)
builder.add_node("classifier_node", classifier_node)
builder.add_node("planning_node", planning_node)
builder.add_node("scene_executor_node", scene_executor_node)
builder.add_node("ffmpeg_node", ffmpeg_node)
builder.add_node("text_node", text_node)

builder.add_edge(START, "classifier_node")
builder.add_conditional_edges(
    "classifier_node",
    route_after_classify,
    {
        "manim_only": "planning_node",
        "text": "text_node",
    },
)

builder.add_edge("planning_node", "scene_executor_node")
builder.add_conditional_edges(
    "scene_executor_node",
    route_after_scene,
    {
        "scene_executor_node": "scene_executor_node",
        "ffmpeg_node": "ffmpeg_node",
    },
)

builder.add_edge("ffmpeg_node", END)
builder.add_edge("text_node", END)
agent = builder.compile()


class Agent:
    def __init__(
        self,
        session_id: str,
        create_video: Optional[bool],
        task_id: Optional[str] = None,
        use_LLM: bool = True,
    ):
        self.session_id = session_id
        self.use_LLM = use_LLM
        self.GEN_DIR = GEN_DIR
        self.MEDIA_DIR = MEDIA_DIR
        self.create_video = create_video
        self.task_id = _build_task_id(task_id)

    def run_agent(self, query: str):
        route = None
        if self.create_video is True:
            route = "manim_only"
        elif self.create_video is False:
            route = "text"

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "llm_calls": 0,
            "scene_plan": [],
            "current_index": 0,
            "completed_urls": [],
            "final_video_url": None,
            "route": route,
            "task_id": self.task_id,
            "session_id": self.session_id,
        }

        try:
            result = agent.invoke(initial_state)
            resolved_route = result.get("route") or route or "text"

            if result.get("final_video_url"):
                return {
                    "string": result["final_video_url"],
                    "ok": True,
                    "route": resolved_route,
                    "task_id": self.task_id,
                }
            if result.get("messages"):
                return {
                    "string": result["messages"][-1].content,
                    "ok": True,
                    "route": resolved_route,
                    "task_id": self.task_id,
                }
            return {
                "string": str(result),
                "ok": True,
                "route": resolved_route,
                "task_id": self.task_id,
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "task_id": self.task_id,
                "route": route or "unknown",
            }
