import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import create_client, Client
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from huggingface_hub import InferenceClient
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
    def __init__(self, embeddings, model, session_id, research=""): # gets the model and the embedding model
        self.embeddings = embeddings
        self.model = model
        self.session_id = session_id
        # self.active_sessions = {"session_id":session_id}
        self.research = research

        self.supabase: Client = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_KEY"]
        )
        self.inference_client = InferenceClient(
            api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )

    def get_user_vector_store(self, session_id: str) -> SupabaseVectorStore:
        # Each session is isolated via metadata filter, not separate collections
        return SupabaseVectorStore(
            client=self.supabase,
            embedding=self.embeddings,
            table_name="documents",
            query_name="match_documents",  # see step 5 below
        )

    def store(self, source, session_id: str, is_text=False, source_label="raw_text"):
        vector_store = self.get_user_vector_store(session_id)

        if is_text:
            documents = [Document(
                page_content=source,
                metadata={'source': source_label, 'session_id': session_id}
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

    def _match_documents_rpc(self, query_embedding, session_id: str, limit: int):
        candidate_payloads = [
            {
                "query_embedding": query_embedding,
                "match_count": limit * 2,
                "filter": {"session_id": session_id},
            },
            {
                "query_embedding": query_embedding,
                "filter": {"session_id": session_id},
            },
            {
                "query_embedding": query_embedding,
                "match_count": limit * 2,
            },
            {
                "query_embedding": query_embedding,
            },
        ]

        last_error = None
        for payload in candidate_payloads:
            try:
                result = self.supabase.rpc("match_documents", payload).execute()
                rows = result.data or []
                rows = [
                    row
                    for row in rows
                    if row.get("metadata", {}).get("session_id") == session_id
                ]
                return rows[:limit]
            except Exception as exc:
                last_error = exc
                print(
                    "match_documents RPC failed for session "
                    f"{session_id} with payload keys {list(payload.keys())}: {exc}"
                )

        if last_error:
            raise RuntimeError(
                f"Unable to retrieve documents from Supabase for session {session_id}"
            ) from last_error
        return []

    def retrieve_similarity(self, query: str, session_id: str):
        query_embedding = self.embeddings.embed_query(query)
        rows = self._match_documents_rpc(
            query_embedding=query_embedding,
            session_id=session_id,
            limit=5,
        )

        docs = [
            Document(
                page_content=row.get("content", ""),
                metadata=row.get("metadata", {}),
            )
            for row in rows
            if row.get("content")
        ]

        serialized = "\n\n".join(
            f"Source: {d.metadata}\nContent: {d.page_content}"
            for d in docs
        )
        return serialized, docs

    def infer(self, query: str, session_id: str):
        try:
            serialized_context, _ = self.retrieve_similarity(query, session_id)
        except Exception as exc:
            print(f"Context retrieval failed for session {session_id}: {exc}")
            serialized_context = ""

        if serialized_context:
            messages = [
                SystemMessage(
                    content=(
                        "You are a retrieval QA assistant. Use the retrieved context as the primary grounding for your answer. "
                        "If the context is incomplete, you may supplement it with general knowledge or additional research, "
                        "but clearly avoid inventing facts."
                        "If there are no research contents found, try answering on your own."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{query}\n\n"
                        f"Retrieved Context:\n{serialized_context}"
                        f"\n\nAdditional Research:\n{self.research}"
                    )
                ),
            ]
        else:
            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful educational assistant. Answer clearly and directly. "
                        "No retrieved session context is available, so use your own knowledge and any additional research provided. "
                        "If something is uncertain, say so briefly."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{query}"
                        f"\n\nAdditional Research:\n{self.research}"
                    )
                ),
            ]

        response = self.model.invoke(messages)
        content = response.content if hasattr(response, "content") else response
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content if item)
        content = str(content).strip()
        if content:
            return {
                "content": content,
                "research": self.research,
            }

        print(
            f"ChatHuggingFace returned empty content for session {session_id}. "
            "Falling back to direct Hugging Face chat completion."
        )
        direct_messages = [
            {
                "role": "system" if isinstance(message, SystemMessage) else "user",
                "content": message.content,
            }
            for message in messages
        ]
        direct_response = self.inference_client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=direct_messages,
            max_tokens=2500,
            temperature=0.3,
        )
        direct_content = direct_response.choices[0].message.content
        direct_content = str(direct_content).strip() if direct_content is not None else ""
        if direct_content:
            return {
                "content": direct_content,
                "research": self.research,
            }
        return {
            "content": "I could not generate a useful answer from the available context and model response.",
            "research": self.research,
        }
        

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
