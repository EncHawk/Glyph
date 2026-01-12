import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

model  = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
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
        chunk_overlap=10,
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
    retrieved_data = vector_store.similarity_search(query,k=5)
    serialized_data = "__--analysis--__".join(
        (f"source:{doc.metadata} \n content : {doc.page_content}") 
        for doc in retrieved_data)
    return serialized_data, retrieved_data

def inferAgent(query: str):
    tools = [retrieve_similarity]
    agent = create_agent(model, tools)
    
    messages = [
        SystemMessage(content=(
            "You have access to a tool that retrieves data from a pdf file that a user has provided. "
            "Use the tool to create meaningful and contextual response to the user's query."
        )),
        {"role": "user", "content": query}
    ]
    result = agent.invoke({"messages": messages})
    return result["messages"][-1]


if __name__=="__main__":
    print('entered rag.py')
    path = os.path.join(os.path.dirname(__name__),'uploads','nasa.pdf')
    text_path = os.path.join(os.path.dirname(__name__),'uplaods','input.txt')
    stores(text_path, isText=True)
    query="""
        Help me understand the general terms of the agreement while youre at it, delve in to this part of the agreement:
        `waiver and indemnity`
    """
    res = inferAgent(query)
    print(res)
# pip install langchain-community pytesseract pdf2image
