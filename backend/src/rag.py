import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_core.documents import Document
from langchain.messages import HumanMessage, SystemMessage
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
    def __init__(self, embeddings, model, session_id): # gets the model and the embedding model
        self.embeddings = embeddings
        self.model = model
        self.session_id = session_id
        # self.active_sessions = {"session_id":session_id}

        self.supabase: Client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"]
        )

    def get_user_vector_store(self, session_id: str) -> SupabaseVectorStore:
        # Each session is isolated via metadata filter, not separate collections
        return SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents",  # see step 5 below
        )

    def store(self, source, session_id: str, is_text=False):
        vector_store = self.get_user_vector_store(session_id)

        if is_text:
            documents = [Document(
                page_content=source,
                metadata={'source': 'raw_text', 'session_id': session_id}
            )]
        else:
            loader = UnstructuredPDFLoader(source)
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
        docs = vector_store.similarity_search(
            query,
            k=5,
            filter={"session_id": session_id}
        )


        serialized = "\n\n".join(
            f"Source: {d.metadata}\nContent: {d.page_content}"
            for d in docs
        )
        return serialized, docs

    def infer(self, query: str, session_id: str):
        serialized_context, _ = self.retrieve_similarity(query, session_id)
        if not serialized_context:
            serialized_context = "No relevant context was found in the vector store for this session."

        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a retrieval QA assistant. Use the retrieved context to answer. "
                        "If context is missing, say that clearly and answer conservatively."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{query}\n\n"
                        f"Retrieved Context:\n{serialized_context}"
                    )
                ),
            ]
        )
        return response.content if hasattr(response, "content") else str(response)
        

# run this while testing.
# if __name__=="__main__":
#     print('entered rag.py')
#     path = os.path.join(os.path.dirname(__name__),'uploads','Jeevan_Koiri_Software_Engineer.pdf')
#     text_path = os.path.join(os.path.dirname(__name__),'uplaods','input.txt')
#     rag = RagAgent() 
#     store(path)
#     query="""
#         im looking for a go developer, can you help me find one? gimme their contact details from

#     """
#     res = inferAgent(query)
    # print(res.content)
# pip install langchain-community pytesseract pdf2image
