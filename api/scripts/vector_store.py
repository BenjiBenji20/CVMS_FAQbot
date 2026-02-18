import json
from uuid import uuid4
from pathlib import Path
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

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

# load the mardown file
def load_markdown_files() -> list[Document]:
    """Load and chunk markdown files by headers"""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    
    chunks = []
    
    for md_file in DOCS_DIR.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # split by headers
        md_chunks = md_splitter.split_text(content)
        
        # add metadata
        for chunk in md_chunks:
            chunk.metadata.update({
                "chunk_id": str(uuid4()),
                "source": md_file.name,
                "doc_type": "faq",
                "type": "knowledge",
                "added_date": datetime.now().isoformat()
            })
            
        chunks.extend(md_chunks)
            
    return chunks

def load_json_files() -> list[Document]:
    """Load data from JSON and convert to embeddable documents"""
    json_files = DOCS_DIR / "cvms-structured-data.json"
    
    if not json_files.exists():
        return []
    
    with open(json_files, 'r', encoding='utf-8') as f:
        json_list = json.load(f)
        
        structured_json_docs = []
        
        for j in json_list:
            # create embeddable text
            keywords = ', '.join(j.get('intent', []))
            text = f"""[ACTION:{j['id']}]
                    Title: {j['title']}
                    Description: {j['description']}
                    Keywords: {keywords}
                    Category: {j.get('category', 'General')}
                    """
                    
            # create the document
            doc = Document(
                page_content=text,
                metadata={
                    'chunk_id': str(uuid4()),
                    'type': 'action',
                    'doc_type': 'faq',
                    'action_id': j['id'],
                    'url': j['url'],
                    'title': j['title'],
                    'button_text': j['button_text'],
                    'source': 'cvms-structured-data.json',
                    'added_date': datetime.now().isoformat()
                }
            )
            structured_json_docs.append(doc)
        
    return structured_json_docs

# Load and index documents
# Load MD files
md_chunks = load_markdown_files()

# Load actions and links in json 
action_chunks = load_json_files()

# Combine
all_docs = md_chunks + action_chunks

# Create UUIDs
uuids = [str(uuid4()) for _ in range(len(all_docs))]

# Store in vector DB
vector_store.add_documents(documents=all_docs, ids=uuids)
