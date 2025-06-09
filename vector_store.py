from pathlib import Path
from typing import List, Optional
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging
import shutil
import time

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage the ChromaDB vector store"""
    
    def __init__(self, vector_store_dir: Path, embedding_model: str, chunk_size: int, chunk_overlap: int):
        self.vector_store_dir = vector_store_dir
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chroma_client = None
        self.collection = None
        self.index = None
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            max_length=512
        )
        Settings.embed_model = self.embed_model
        
        # Initialize node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def close(self):
        """Close ChromaDB client connections"""
        if self.chroma_client:
            try:
                # ChromaDB doesn't have an explicit close method, but we can reset the client
                self.chroma_client = None
                self.collection = None
                # Give it a moment to release file locks
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")
    
    def create_index(self, documents: List, overwrite: bool = False):
        """Create vector index from documents"""
        if self.vector_store_dir.exists() and not overwrite:
            logger.info("Vector store already exists. Use overwrite=True to recreate.")
            return False
        
        if overwrite and self.vector_store_dir.exists():
            # Close any existing connections first
            self.close()
            
            # Try to remove the directory
            try:
                shutil.rmtree(self.vector_store_dir)
                logger.info("Removed existing vector store")
            except PermissionError:
                # On Windows, sometimes files are locked. Try alternative approach
                logger.warning("Could not remove vector store directory, trying alternative approach")
                import os
                # Close all file handles and retry
                import gc
                gc.collect()
                time.sleep(1)
                try:
                    shutil.rmtree(self.vector_store_dir)
                except Exception as e:
                    logger.error(f"Failed to remove vector store: {e}")
                    logger.info("Creating new collection in existing database")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.vector_store_dir))
        
        # Try to delete existing collection if it exists
        try:
            self.chroma_client.delete_collection("documents")
        except Exception:
            pass  # Collection doesn't exist, that's fine
        
        self.collection = self.chroma_client.create_collection("documents")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        logger.info(f"Creating index from {len(documents)} documents...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info("Vector store created successfully")
        return True
    
    def load_index(self):
        """Load existing vector index"""
        if not self.vector_store_dir.exists():
            logger.error("Vector store directory does not exist")
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
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the vector index"""
        return self.index
