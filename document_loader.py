import os
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document
import pypdf
import chardet
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load documents from the archive folder"""
    
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf', '.md', '.html', '.csv'}
    
    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
    
    def load_all_documents(self) -> List[Document]:
        """Load all documents from the archive directory"""
        documents = []
        
        for file_path in self.archive_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load_document(file_path)
                    if doc:
                        documents.append(doc)
                        logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def load_document(self, file_path: Path) -> Document:
        """Load a single document based on its type"""
        if file_path.suffix.lower() == '.pdf':
            return self._load_pdf(file_path)
        elif file_path.suffix.lower() in {'.txt', '.md', '.html', '.csv'}:
            return self._load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    def _load_pdf(self, file_path: Path) -> Document:
        """Load PDF document"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        metadata = {
            "file_name": file_path.name,
            "file_type": "pdf",
            "file_path": str(file_path)
        }
        
        return Document(text=text.strip(), metadata=metadata)
    
    def _load_text(self, file_path: Path) -> Document:
        """Load text-based document with encoding detection"""
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] or 'utf-8'
        
        # Read with detected encoding
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            text = file.read()
        
        metadata = {
            "file_name": file_path.name,
            "file_type": file_path.suffix[1:],
            "file_path": str(file_path)
        }
        
        return Document(text=text.strip(), metadata=metadata)
        