from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Optional, Any
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import uuid
from datetime import datetime
import logging
from fastapi.middleware.cors import CORSMiddleware
import json
import faiss
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class URLRequest(BaseModel):
    url: HttpUrl
    description: Optional[str] = ""

class URLResponse(BaseModel):
    url: str
    embedding_id: str
    timestamp: str
    description: str
    status: str

class QueryRequest(BaseModel):
    embedding_id: str
    query: str
    k: int = 3

class MultiQueryRequest(BaseModel):
    embedding_ids: List[str]
    query: str
    k: int = 3

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# In-memory storage
url_embeddings: Dict[str, Dict] = {}

# Helper functions
def process_url_content(url: str) -> tuple[FAISS, int]:
    """Process URL content and return FAISS index and chunk count."""
    try:
        # Load content
        loader = WebBaseLoader(str(url))
        documents = loader.load()
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create FAISS index
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        return vectorstore, len(chunks)
    
    except Exception as e:
        logger.error(f"Error processing URL {url}: {str(e)}")
        raise

def save_vectorstore(vectorstore: FAISS, embedding_id: str):
    """Save FAISS index to disk."""
    try:
        save_dir = f"indexes/{embedding_id}"
        os.makedirs(save_dir, exist_ok=True)
        vectorstore.save_local(save_dir)
    except Exception as e:
        logger.error(f"Error saving vectorstore: {str(e)}")
        raise

def load_vectorstore(embedding_id: str) -> Optional[FAISS]:
    """Load FAISS index from disk."""
    try:
        save_dir = f"indexes/{embedding_id}"
        if os.path.exists(save_dir):
            return FAISS.load_local(save_dir, embeddings)
        return None
    except Exception as e:
        logger.error(f"Error loading vectorstore: {str(e)}")
        raise

# API Endpoints
@app.post("/process-url", response_model=URLResponse)
async def process_url(request: URLRequest, background_tasks: BackgroundTasks):
    try:
        # Generate unique ID
        embedding_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Process URL
        vectorstore, chunk_count = process_url_content(str(request.url))
        
        # Store in memory
        url_embeddings[embedding_id] = {
            'vectorstore': vectorstore,
            'url': str(request.url),
            'timestamp': timestamp,
            'description': request.description,
            'chunk_count': chunk_count
        }
        
        # Save to disk in background
        background_tasks.add_task(save_vectorstore, vectorstore, embedding_id)
        
        return URLResponse(
            url=str(request.url),
            embedding_id=embedding_id,
            timestamp=timestamp,
            description=request.description,
            status="complete"
        )
    
    except Exception as e:
        logger.error(f"Error in process_url: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_embeddings(request: QueryRequest):
    try:
        # Load vectorstore if not in memory
        if request.embedding_id not in url_embeddings:
            vectorstore = load_vectorstore(request.embedding_id)
            if not vectorstore:
                raise HTTPException(
                    status_code=404,
                    detail="Embedding ID not found"
                )
            url_embeddings[request.embedding_id] = {
                'vectorstore': vectorstore
            }
        
        vectorstore = url_embeddings[request.embedding_id]['vectorstore']
        results = vectorstore.similarity_search(
            request.query,
            k=request.k
        )
        
        return {
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in query_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-multiple")
async def query_multiple_embeddings(request: MultiQueryRequest):
    try:
        all_results = []
        
        # Validate embedding IDs
        missing_ids = [
            id for id in request.embedding_ids 
            if id not in url_embeddings and not load_vectorstore(id)
        ]
        if missing_ids:
            raise HTTPException(
                status_code=404,
                detail=f"Embedding IDs not found: {missing_ids}"
            )
        
        # Query each vectorstore
        for embedding_id in request.embedding_ids:
            # Load if not in memory
            if embedding_id not in url_embeddings:
                vectorstore = load_vectorstore(embedding_id)
                url_embeddings[embedding_id] = {
                    'vectorstore': vectorstore
                }
            
            vectorstore = url_embeddings[embedding_id]['vectorstore']
            results = vectorstore.similarity_search(
                request.query,
                k=request.k
            )
            
            # Add source information
            results_with_source = [
                {
                    "content": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "embedding_id": embedding_id
                    }
                }
                for doc in results
            ]
            all_results.extend(results_with_source)
        
        return {
            "results": all_results
        }
    
    except Exception as e:
        logger.error(f"Error in query_multiple_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-embeddings")
async def list_embeddings():
    try:
        # List both in-memory and saved embeddings
        all_embeddings = {}
        
        # Add in-memory embeddings
        for id, data in url_embeddings.items():
            all_embeddings[id] = {
                'url': data.get('url', 'Unknown'),
                'timestamp': data.get('timestamp', 'Unknown'),
                'description': data.get('description', '')
            }
        
        # Check disk for additional embeddings
        if os.path.exists('indexes'):
            for embedding_id in os.listdir('indexes'):
                if embedding_id not in all_embeddings:
                    vectorstore = load_vectorstore(embedding_id)
                    if vectorstore:
                        all_embeddings[embedding_id] = {
                            'url': 'Loaded from disk',
                            'timestamp': 'Unknown',
                            'description': 'Loaded from disk'
                        }
        
        return all_embeddings
    
    except Exception as e:
        logger.error(f"Error in list_embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Create indexes directory
    os.makedirs("indexes", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
