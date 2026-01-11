import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

model  = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store= Chroma(
    collection_name="glyph-rag",
    embedding_function = embeddings,
    persist_directory = './glyphDB'
)


def stores(filename):
    uploads = f"./reads/{filename}"
    print(uploads)