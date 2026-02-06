from uuid import uuid4
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.config.settings import settings

THIS_FILE_DIR = Path(__file__).parent
DOCS_DIR = THIS_FILE_DIR.parent / "documents"
PERSISTENT_CHROMADB = THIS_FILE_DIR.parent / "chroma_db"

# initiate embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    api_key=settings.EMBEDDING_MODEL_API_KEY,
    model=settings.MODEL_NAME
)

# specify a directory for persistence embeddings 
vector_store = Chroma(
    collection_name="cvms_doc_collections",
    embedding_function=embedding_model,
    persist_directory=str(PERSISTENT_CHROMADB)
)

# load the pdf docs
loader = PyPDFDirectoryLoader(str(DOCS_DIR))

# raw document
raw_doc = loader.load()

# configure doc text splitter
doc_text_splitter =  RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try paragraph, then sentence, then word breaks
)

# chunks the raw documents based on splitter configuration
chunks = doc_text_splitter.split_documents(raw_doc)

# create ids for every chunks of documents
uuids = [str(uuid4()) for _ in range(len(chunks))]

# store the chunks in vector/memory
vector_store.add_documents(
    documents=chunks,
    ids=uuids
)
