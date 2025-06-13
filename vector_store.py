from pathlib import Path
from typing import List, Optional, Set
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
import logging
import shutil
import time
import json

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage the ChromaDB vector store with document tracking"""
    
    def __init__(self, vector_store_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int):
        self.vector_store_dir = vector_store_dir
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_client = None
        self.collection = None
        self.index = None
        self.indexed_docs_file = vector_store_dir / "indexed_documents.json"
        
        # Initialize embedding model
        logger.info(f"Initializing embedding model: {embedding_model}")
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            max_length=512,
            embed_batch_size=32
        )
        Settings.embed_model = self.embed_model
        
        # Initialize node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def exists(self) -> bool:
        """Check if vector store exists"""
        return self.vector_store_dir.exists() and self.indexed_docs_file.exists()
    
    def get_indexed_documents(self) -> Set[str]:
        """Get set of indexed document names"""
        if not self.indexed_docs_file.exists():
            return set()
        
        try:
            with open(self.indexed_docs_file, 'r') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Error reading indexed documents: {e}")
            return set()
    
    def _save_indexed_documents(self, doc_names: Set[str]):
        """Save the set of indexed document names"""
        try:
            self.vector_store_dir.mkdir(exist_ok=True)
            with open(self.indexed_docs_file, 'w') as f:
                json.dump(list(doc_names), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving indexed documents: {e}")
    
    def close(self):
        """Close ChromaDB client connections"""
        if self.chroma_client:
            try:
                self.chroma_client = None
                self.collection = None
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")
    
    def create_index(self, documents: List[Document], overwrite: bool = False) -> bool:
        try:
            if self.exists() and overwrite:
                print("♻️ Removing existing vector store...")
                self.close()
                shutil.rmtree(self.vector_store_dir, ignore_errors=True)
            
            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            
            # Create new collection
            self.collection = self.chroma_client.get_or_create_collection("documents")
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with progress
            print("🔧 Building vector index...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            # Save indexed documents
            doc_names = {doc.metadata.get('file_name', f'doc_{i}') 
                        for i, doc in enumerate(documents)}
            self._save_indexed_documents(doc_names)
            
            print("✅ Vector store created")
            return True
        except Exception as e:
            logger.exception("Vector store creation failed")
            return False
    
    def load_index(self) -> bool:
        """Load existing vector index"""
        if not self.exists():
            logger.error("Vector store does not exist")
            return False
        
        try:
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            
            # Check if collection exists
            collections = self.chroma_client.list_collections()
            collection_names = [col.name for col in collections]
            
            if "documents" not in collection_names:
                logger.error("Collection 'documents' not found in vector store")
                return False
            
            self.collection = self.chroma_client.get_collection("documents")
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=self.collection)
            
            # Create index from existing vector store
            self.index = VectorStoreIndex.from_vector_store(vector_store)
            
            logger.info("Vector store loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add new documents to existing vector store"""
        if not self.exists():
            logger.error("Vector store does not exist")
            return False
        
        try:
            # Load index if not already loaded
            if not self.index:
                if not self.load_index():
                    return False
            
            # Get current indexed documents
            indexed_docs = self.get_indexed_documents()
            
            # Add each document
            for doc in documents:
                try:
                    # Insert into index
                    self.index.insert(doc)
                    
                    # Track document name
                    doc_name = doc.metadata.get('file_name', 'unknown')
                    indexed_docs.add(doc_name)
                    logger.info(f"Added document: {doc_name}")
                    
                except Exception as e:
                    logger.error(f"Error adding document: {e}")
                    continue
            
            # Save updated indexed documents list
            self._save_indexed_documents(indexed_docs)
            
            logger.info(f"Successfully added {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector index"""
        if not self.index and self.exists():
            self.load_index()
        return self.index
        