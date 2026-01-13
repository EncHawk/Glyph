import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint( # add your huggingface token, this shit free and good heck yeah!
    repo_id= "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    temperature = 0.7,
)
model = ChatHuggingFace(llm=llm)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store= Chroma(
    collection_name="glyph-rag",
    embedding_function = embeddings,
    persist_directory = './glyphDB'
)


def stores(source, isText=False):
    """
        FILEPATH IS BEING USED AS AN ALIAS FOR THE TEXT PART OF THE FUNCTION.
        FIX: add the chaneges to make this compatible to by naming convention
        takes an input string, to produce a boolean to validate
        whether or not storing the embeddings worked. 
        
        ::server should call the LLM next with the embeddings.
    """

    # ai fix, instead of repeating stuff for both pdf and text use a common documents var.
    if not isText:
        # uploads = f"./reads/{filename}"
        loader = PyPDFLoader(source)
        documents = loader.load()
        metadata_source= source
    else:
        documents= [ 
            Document(
                page_content= source,
                metadata = {'source':"raw_text"}
            )
        ]
        metadata_source = 'raw_text'

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"the number of splits {len(splits)}")
    document_ids = vector_store.add_documents(splits)
    print(document_ids[:3])

@tool(response_format="content_and_artifact")
def retrieve_similarity(query:str): # for a query searches the related documents
    """ returns a list of similar indices in the vector store that match the query in terms of some similarity"""
    retrieved_docs = vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def inferAgent(query: str):
    tools = [retrieve_similarity]
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries in the name of retrieve similarity, always use this before generating any response",
        "NO MATTER WHAT THE USER RESPONSE IS, YOU MUST ALWAYS REFER ONLY TO THE OUTPUT FROM THE TOOL AND NOTHING ELSE, ALL YOUR INTELLIGENCE IS RETRIEVED FROM THE TOOL'S OUTPUT"
    )
    agent = create_agent(model, tools, system_prompt= prompt)
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


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
