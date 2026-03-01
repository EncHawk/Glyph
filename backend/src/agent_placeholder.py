from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain.tools import tool
from pydantic import constr, BaseModel
import os
import subprocess
import sys
import datetime
import boto3
from langchain.messages import SystemMessage, AnyMessage, HumanMessage, ToolMessage
import botocore
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AnyMessage
from langchain.messages import ToolMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal, List, Optional
import json
from rag import RagAgent
import operator





llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)
model = ChatHuggingFace(llm=llm)

def manim_response(self, className: str, s3_bucket_name: str):
    os.makedirs(self.GEN_DIR, exist_ok=True)
    os.makedirs(self.MEDIA_DIR, exist_ok=True)
    assert os.path.join(self.MEDIA_DIR, "videos", "480p15", f"{className}.mp4")
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("aws_region"),
    )
    output_vid = os.path.abspath(os.path.join(self.GEN_DIR, f"{className}.py"))
    generated_video = os.path.join(
        self.MEDIA_DIR, "videos", className, "480p15", f"{className}.mp4"
    )
    try:
        subprocess.run(["manim", "-pql", output_vid, className], check=True)
    except Exception as e:
        print("Manim died:", e)
        raise Exception("Manim render failed")
    s3_key = f"manim/{className}_{datetime.datetime.now().isoformat()}.mp4"
    try:
        s3.Bucket(s3_bucket_name).upload_file(
            generated_video, s3_key
        )  # ffs aws was getting a python file in the format of mp4 im retarded
    except botocore.exceptions.ClientError as e:
        print("S3 upload failed:", e)
        raise Exception("S3 upload failed")
    s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_key}"
    return s3_url


@tool
def manim_tool(self, prompt, correction_context=None):
    """
    :param prompt: the input prompt to generate the illustration code
    :param correction_context: Optional dict with previous error and code for retry
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

    try:
        # If we have correction context, modify the prompt
        if correction_context:
            enhanced_prompt = f"""
                {prompt}

                IMPORTANT: Previous attempt failed with this error:
                {correction_context["error"]}

                Previous code that failed:
                ```python
                {correction_context["code"]}
                ```

                Fix the error and generate corrected code. Common Manim fixes:
                - Don't use 'opacity' parameter in Dot() - use set_opacity() method instead
                - Use fill_opacity and stroke_opacity for VMobjects
                - Ensure all imports are correct
                - Check method signatures match Manim v0.19.2 API
            """
            instance = Manim(
                response_format=response_format, prompt=enhanced_prompt
            )
        else:
            instance = Manim(response_format=response_format, prompt=prompt)

        response_text = instance.inferModel(model=ModelResponse)

        # Ensure the directory exists
        os.makedirs(self.GEN_DIR, exist_ok=True)

        # Write the generated code to file
        file = os.path.join(self.GEN_DIR, f"{response_text.className}.py")
        with open(file, "w") as f:
            f.write(response_text.code)

        subprocess.run(["manim", "-pql", file, response_text.className], check=True)
        aws_string = self.manim_response(
            response_text.className, "glyph-data-storage"
        )
        return {
            "ok": True,
            "msg": "File Ran successfully",
            "string": aws_string,
            "code": response_text.code,  # ← Return code on success too
            "className": response_text.className,
        }
    except Exception as e:
        # Try to return the code even on failure
        error_code = None
        class_name = None
        try:
            error_code = response_text.code if "response_text" in locals() else None
            class_name = (
                response_text.className if "response_text" in locals() else None
            )
        except:
            pass

        return {
            "ok": False,
            "error": str(e),
            "code": error_code,  # ← Return code that failed
            "className": class_name,
        }
    
@tool
def contextual_text_tool(prompt):
    """
    Docstring for flowchart

    :param prompt: the prompt to generate the code for graphviz flowchart
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
    llm = HuggingFaceEndpoint(  # add your huggingface token, this shit free and good heck yeah!
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.7,
    )
    model = ChatHuggingFace(llm=llm)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    instance = RagAgent(embeddings=embeddings, model=model)
    response_code = instance.inferModel(input=prompt)
    return response_code

class Scene(TypedDict):
    id: str
    type: str          # "manim" | "diffusion"
    class_name: str
    prompt: str
    s3_url: Optional[str]

class AgentState(TypedDict):
    messages:        Annotated[list[AnyMessage], operator.add]
    llm_calls:       int
    scene_plan:      List[Scene]          # filled by planner
    current_index:   int                  # which scene we're rendering
    completed_urls:  Annotated[list[str], operator.add]  # s3 urls in order
    final_video_url: Optional[str]
    route:           str                  # "manim_only" | "manim_diffusion" | "text"

tools = [manim_tool]
tools_by_name = {t.name: t for t in tools}
model_with_tools = ChatHuggingFace(llm=llm).bind_tools(tools=tools)



def classifier_node(state: AgentState) -> AgentState:
    """Classifies user intent → sets state['route']"""
    user_msg = state["messages"][-1].content
    prompt = f"""You are a router. Given the user message, output ONLY one of:
- manim_only       → if they want animated math/code visuals
- manim_diffusion  → if they want animated visuals + photorealistic images
- text             → if they just want a text explanation

User: {user_msg}
Route:"""
    route = model.invoke([SystemMessage(content=prompt)]).content.strip().lower()
    if route not in ("manim_only", "manim_diffusion", "text"):
        route = "text"
    return {"route": route, "llm_calls": state.get("llm_calls", 0) + 1}


def planning_node(state: AgentState) -> AgentState:
    """Plans the scene list as structured JSON based on user prompt."""
    user_msg = state["messages"][-1].content
    prompt = f"""You are a video scene planner for an educational explainer video.
Given the topic below, output ONLY a valid JSON array of scenes.

Rules:
- Each scene has: id (scene_01, scene_02...), type ("manim"), class_name (PascalCase), prompt (detailed instruction for that scene)
- Start with an IntroductionScene, end with a WrapUpScene
- 3-6 scenes total
- No markdown, no explanation, raw JSON only

Topic: {user_msg}

Example output:
[
  {{"id":"scene_01","type":"manim","class_name":"IntroductionScene","prompt":"Animate the title and a brief overview"}},
  {{"id":"scene_02","type":"manim","class_name":"CoreConceptScene","prompt":"Show the core concept with step-by-step animation"}}
]"""

    response = model.invoke([SystemMessage(content=prompt)]).content.strip()

    # strip markdown fences if model wraps it anyway
    response = response.replace("```json", "").replace("```", "").strip()
    scene_plan = json.loads(response)

    # inject s3_url=None placeholder
    for scene in scene_plan:
        scene.setdefault("s3_url", None)

    return {
        "scene_plan": scene_plan,
        "current_index": 0,
        "completed_urls": [],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def scene_executor_node(state: AgentState) -> AgentState:
    """Picks the current scene from plan and calls manim_tool."""
    idx = state["current_index"]
    scene = state["scene_plan"][idx]

    result = manim_tool.invoke({
        "prompt": scene["prompt"],
        "correction_context": None
    })

    s3_url = result.get("string") if result.get("ok") else f"ERROR:{result.get('error')}"

    # update plan with s3_url
    updated_plan = list(state["scene_plan"])
    updated_plan[idx] = {**scene, "s3_url": s3_url}

    return {
        "scene_plan": updated_plan,
        "current_index": idx + 1,
        "completed_urls": [s3_url],
    }


def ffmpeg_node(state: AgentState) -> AgentState:
    """Downloads all S3 videos, stitches with ffmpeg, re-uploads."""
    urls = state["completed_urls"]
    local_files = []

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("aws_access_key_id"),
        aws_secret_access_key=os.getenv("aws_secret_access_key"),
        region_name=os.getenv("aws_region"),
    )
    bucket = "glyph-data-storage"

    for i, url in enumerate(urls):
        local_path = f"/tmp/scene_{i:02d}.mp4"
        s3_key = url.split(f"{bucket}.s3.amazonaws.com/")[1]
        s3.download_file(bucket, s3_key, local_path)
        local_files.append(local_path)

    # write ffmpeg concat list
    concat_path = "/tmp/concat_list.txt"
    with open(concat_path, "w") as f:
        for fp in local_files:
            f.write(f"file '{fp}'\n")

    output_path = f"/tmp/final_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    subprocess.run(
        ["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_path, "-c", "copy", output_path],
        check=True
    )

    # upload final
    final_key = f"final/{os.path.basename(output_path)}"
    s3.upload_file(output_path, bucket, final_key)
    final_url = f"https://{bucket}.s3.amazonaws.com/{final_key}"

    return {
        "final_video_url": final_url,
        "messages": [HumanMessage(content=f"Final video ready: {final_url}")]
    }


def text_node(state: AgentState) -> AgentState:
    """Falls through to contextual_text_tool for pure text answers."""
    result = contextual_text_tool.invoke({"prompt": state["messages"][-1].content})
    return {"messages": [HumanMessage(content=str(result))]}


# ── EDGES / CONDITIONS ────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    return state["route"]   # → "manim_only" | "manim_diffusion" | "text"

def route_after_scene(state: AgentState) -> str:
    """Loop back if more scenes remain, else go to ffmpeg."""
    if state["current_index"] < len(state["scene_plan"]):
        return "scene_executor_node"
    return "ffmpeg_node"

# ── GRAPH ─────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("classifier_node",   classifier_node)
builder.add_node("planning_node",     planning_node)
builder.add_node("scene_executor_node", scene_executor_node)
builder.add_node("ffmpeg_node",       ffmpeg_node)
builder.add_node("text_node",         text_node)

builder.add_edge(START, "classifier_node")

builder.add_conditional_edges("classifier_node", route_after_classify, {
    "manim_only":      "planning_node",
    "text":            "text_node",
})

builder.add_edge("planning_node", "scene_executor_node")

builder.add_conditional_edges("scene_executor_node", route_after_scene, {
    "scene_executor_node": "scene_executor_node",  # loop
    "ffmpeg_node":         "ffmpeg_node",
})

builder.add_edge("ffmpeg_node", END)
builder.add_edge("text_node",   END)

agent = builder.compile()