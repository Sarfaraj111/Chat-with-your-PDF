#load pdf 
#split into chunks 
#create the embeddings 
#store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma 
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("document_loaders/GRU.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

vectorstore = Chroma.from_documents(
    documents= chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)