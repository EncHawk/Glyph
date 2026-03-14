## GLYPHAI - Multimodal Agentic Chat Platform

Glyph is an open-source platform that enables AI-powered multimodal interactions through standalone tools and an intelligent agentic orchestration layer.

### Features

- **Manim Integration** - Generate mathematical animations and visualizations from natural language prompts using the Manim library
- **RAG (Retrieval Augmented Generation)** - Upload and query documents with semantic search powered by Supabase vector store and HuggingFace embeddings
- **Flowchart Generation** - Create visual flowcharts and diagrams automatically
- **Agentic Orchestration** - LangChain-based agent that intelligently routes requests to the appropriate tool

### Architecture

- **Backend**: Python with Flask, LangChain, HuggingFace endpoints
- **LLM Models**: GLM-4.7 (code generation), Zephyr-7B-beta (basic text queries)
- **Vector Store**: Supabase for document embeddings
- **Rendering**: Manim for video generation, Mermaid for flowcharts
- **Storage**: AWS S3 for generated media files

### Tech Stack

- Python > 3.11
- LangChain & LangChain-HuggingFace
- Langgraph (Agentic orchestration)
- Flask
- HuggingFace Inference API
- Supabase (vector database)
- AWS S3
- Manim

### LangGraph Orchestration

The backend agent uses a LangGraph `StateGraph` to route each request through a small execution graph instead of a single monolithic handler.

- `classifier_node` decides whether the request should go to text generation or Manim video generation.
- `planning_node` creates a normalized scene plan for video requests.
- `scene_executor_node` generates, renders, and uploads one scene at a time.
- `ffmpeg_node` downloads completed scene videos, concatenates them, and uploads the final stitched output to AWS S3.
- `text_node` runs the RAG text path and preserves both generated content and research metadata in the final response.

### Getting Started

```bash
# Backend setup
cd backend
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the server
python src/main.py
```

### Environment Variables

- `HUGGINGFACEHUB_API_TOKEN` - HuggingFace API key
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase API key
- `S3_STORAGE_LINK`: Optional - URL for the AWS S3 bucket.
- `aws_access_key_id` - AWS access key
- `aws_secret_access_key` - AWS secret key
- `aws_region` - AWS region
- `PARALLEL_API_KEY` - Parallel-AI's api key 
