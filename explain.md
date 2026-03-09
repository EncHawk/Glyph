# Agent Pipeline Explanation

This document explains the LangGraph agent pipeline in `agent_placeholder.py` and how each node works together.

## Overview

The agent uses a StateGraph with 5 main nodes and 2 conditional edges to route user requests through the appropriate processing path.

```
START → classifier_node → [planning_node → scene_executor_node → ffmpeg_node] OR [text_node] → END
```

---

## Nodes

### 1. `classifier_node`

**Purpose**: Classifies user intent to determine the processing route.

**Function**: `classifier_node(state: AgentState) -> AgentState`

**Logic**:
- Takes the user's last message
- Invokes the LLM to classify into one of three routes:
  - `manim_only` → User wants animated math/code visuals
  - `manim_diffusion` → User wants animated visuals + photorealistic images (not fully implemented)
  - `text` → User just wants a text explanation

**Sets state**: `route`

---

### 2. `planning_node`

**Purpose**: Creates a structured scene plan for video generation.

**Function**: `planning_node(state: AgentState) -> AgentState`

**Logic**:
- Invokes the LLM to generate a JSON array of scenes
- Each scene contains: `id`, `type`, `class_name`, `prompt`, `s3_url`
- Rules:
  - Must start with `IntroductionScene`
  - Must end with `WrapUpScene`
  - 3-6 scenes total

**Sets state**: `scene_plan`, `current_index`, `completed_urls`

---

### 3. `scene_executor_node`

**Purpose**: Executes the current scene by calling the Manim tool.

**Function**: `scene_executor_node(state: AgentState) -> AgentState`

**Logic**:
- Gets the current scene from `scene_plan` using `current_index`
- Calls `manim_tool.invoke()` with the scene's prompt
- Uploads the rendered video to S3
- Updates the scene plan with the S3 URL

**Tools called**: `manim_tool`

**Sets state**: `scene_plan`, `current_index`, `completed_urls`

---

### 4. `ffmpeg_node`

**Purpose**: Stitches all rendered scene videos into a single final video.

**Function**: `ffmpeg_node(state: AgentState) -> AgentState`

**Logic**:
- Downloads all S3 videos from `completed_urls`
- Creates a concat list file for ffmpeg
- Runs ffmpeg to merge all clips
- Uploads the final video to S3

**Tools called**: None (direct S3/ffmpeg)

**Sets state**: `final_video_url`, adds message to `messages`

---

### 5. `text_node`

**Purpose**: Handles pure text responses (no video generation).

**Function**: `text_node(state: AgentState) -> AgentState`

**Logic**:
- Calls `contextual_text_tool` with the user's prompt
- Returns text response

**Tools called**: `contextual_text_tool` → `RagAgent`

**Sets state**: `messages`

---

## Tools

### `manim_tool`

**Purpose**: Generates and renders Manim animations.

**Function**: `manim_tool(prompt, correction_context=None)`

**Process**:
1. Uses LLM (Qwen2.5-7B-Instruct) to generate Manim code
2. Writes code to `generated-scripts/{className}.py`
3. Runs `manim -pql` to render the video
4. Uploads to S3 bucket `glyph-data-storage`
5. Returns S3 URL

**Returns**: `{, msg, stringok, code, className}`

---

### `contextual_text_tool`

**Purpose**: Generates text responses using RAG.

**Function**: `contextual_text_tool(prompt)`

**Process**:
1. Initializes `RagAgent` with embeddings and model
2. Calls `RagAgent.inferModel()` to generate response
3. Returns text response

**Tools called**: `RagAgent` (from rag.py)

**Returns**: Text response string

---

## Conditional Edges

### `route_after_classify`

Routes based on `state['route']`:
- `manim_only` → `planning_node`
- `text` → `text_node`
- `manim_diffusion` → currently routes to `planning_node` (partial implementation)

### `route_after_scene`

Loops through scenes:
- If `current_index < len(scene_plan)` → loop back to `scene_executor_node`
- Otherwise → `ffmpeg_node`

---

## Agent State Schema

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # Chat history
    llm_calls: int                    # Count of LLM invocations
    scene_plan: List[Scene]           # Planned scenes for video
    current_index: int                 # Which scene is being rendered
    completed_urls: Annotated[list[str], operator.add]  # S3 URLs of rendered scenes
    final_video_url: Optional[str]    # Final stitched video URL
    route: str                         # "manim_only" | "manim_diffusion" | "text"
```
