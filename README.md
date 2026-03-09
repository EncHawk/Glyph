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
- Langgraph (manim video generation)
- Flask
- HuggingFace Inference API
- Supabase (vector database)
- AWS S3
- Manim

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
- `aws_access_key_id` - AWS access key
- `aws_secret_access_key` - AWS secret key
- `aws_region` - AWS region

### License
Proudly opensource, gotta add MIT tho.