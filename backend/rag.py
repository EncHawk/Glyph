import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

model  = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store= Chroma(
    collection_name="glyph-rag",
    embedding_function = embeddings,
    persist_directory = './glyphDB'
)


def stores(filepath):
    """
        takes an input string, to produce a boolean to validate
        whether or not storing the embeddings worked. 
        
        ::server should call the LLM next with the embeddings.
    """
    # uploads = f"./reads/{filename}"
    loader = PyPDFLoader(filepath)
    documents = loader.load()
    # print(documents[0].page_content) # sanity check to see its existense
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    print(f"the number of splits {len(splits)}")
    document_ids = vector_store.add_documents(documents=splits)
    print(document_ids[:3])


if __name__=="__main__":
    print('entered rag.py')
    path = os.path.join(os.path.dirname(__name__),'uploads','resume.pdf')
    stores(path)
# pip install langchain-community pytesseract pdf2image
