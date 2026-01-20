import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pdf2image import convert_from_path
import pytesseract
import uuid
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# move all this into the main endpoint for rag, also ensure that login stuff is included in flask.
# now you'll realise how difficult it is to do this without express.

# llm = HuggingFaceEndpoint( # add your huggingface token, this shit free and good heck yeah!
#     repo_id= "Qwen/Qwen2.5-7B-Instruct",
#     temperature = 0.7,
# )
# model = ChatHuggingFace(llm=llm)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

class RagAgent:
    def __init__(self, embeddings, model):
        self.embeddings = embeddings
        self.model = model
        self.active_sessions = {}

    def get_user_vector_store(self, session_id: str):
        if session_id not in self.active_sessions:
            persist_dir = f'./glyphDB/session_{session_id}'
            self.active_sessions[session_id] = Chroma(
                collection_name=f"session_{session_id}",
                embedding_function=self.embeddings,
                persist_directory=persist_dir
            )
        return self.active_sessions[session_id]

    def store(self, source, session_id: str, is_text=False):
        vector_store = self.get_user_vector_store(session_id)

        if os.listdir(source) == []:
            documents = [Document(
                page_content=source,
                metadata={'source': 'raw_text', 'session_id': session_id}
            )]
        else:
            loader = PyPDFLoader(source)
            documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = splitter.split_documents(documents)
        for s in splits:
            s.metadata['session_id'] = session_id

        return vector_store.add_documents(splits)

    def retrieve_similarity(self, query: str, session_id: str):
        vector_store = self.get_user_vector_store(session_id)
        docs = vector_store.similarity_search(query, k=5)

        serialized = "\n\n".join(
            f"Source: {d.metadata}\nContent: {d.page_content}"
            for d in docs
        )
        return serialized, docs

    def infer(self, query: str, session_id: str):
        tools = [lambda q: self.retrieve_similarity(q, session_id)]

        prompt = "Use the retrieval tool to answer queries for this session."

        agent = create_agent(self.model, tools, system_prompt=prompt)

        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values"
        ):
            event["messages"][-1].pretty_print()

    def cleanup_session(self, session_id: str):
        if session_id in self.active_sessions:
            import shutil
            path = f'./glyphDB/session_{session_id}'
            if os.path.exists(path):
                shutil.rmtree(path)
            del self.active_sessions[session_id]


if __name__=="__main__":
    print('entered rag.py')
    path = os.path.join(os.path.dirname(__name__),'uploads','Jeevan_Koiri_Software_Engineer.pdf')
    text_path = os.path.join(os.path.dirname(__name__),'uplaods','input.txt')
    stores(path)
    query="""
        im looking for a go developer, can you help me find one? gimme their contact details from

    """
    res = inferAgent(query)
    # print(res.content)
# pip install langchain-community pytesseract pdf2image
