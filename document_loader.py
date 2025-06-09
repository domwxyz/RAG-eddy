from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import Document
from llama_index.readers.file import (
    PDFReader,
    UnstructuredReader,
    SimpleDirectoryReader
)
import logging
import chardet

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load documents from various file formats"""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.html', '.csv'}
    
    def __init__(self, archive_dir: Path):
        self.archive_dir = archive_dir
        self.pdf_reader = PDFReader()
        
    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding for text files"""
        try:
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))  # Read first 10KB
                encoding = result['encoding']
                confidence = result['confidence']
                
                if confidence > 0.7 and encoding:
                    return encoding
                else:
                    return 'utf-8'  # Default fallback
        except Exception:
            return 'utf-8'
    
    def load_text_file(self, file_path: Path) -> List[Document]:
        """Load a text file with encoding detection"""
        try:
            # Detect encoding
            encoding = self.detect_encoding(file_path)
            
            # Read file
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Create document
            metadata = {
                'file_name': file_path.name,
                'file_type': file_path.suffix,
                'encoding': encoding
            }
            
            return [Document(text=content, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return []
    
    def load_pdf_file(self, file_path: Path) -> List[Document]:
        """Load a PDF file"""
        try:
            documents = self.pdf_reader.load_data(file=file_path)
            
            # Add custom metadata
            for doc in documents:
                doc.metadata.update({
                    'file_name': file_path.name,
                    'file_type': '.pdf'
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return []
    
    def load_csv_file(self, file_path: Path) -> List[Document]:
        """Load a CSV file as a document"""
        try:
            # Detect encoding
            encoding = self.detect_encoding(file_path)
            
            # Read CSV content
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Create document with CSV content
            metadata = {
                'file_name': file_path.name,
                'file_type': '.csv',
                'encoding': encoding
            }
            
            # Add a note that this is tabular data
            doc_text = f"CSV File: {file_path.name}\n\n{content}"
            
            return [Document(text=doc_text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return []
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load a single document based on its type"""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        suffix = file_path.suffix.lower()
        
        if suffix not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {suffix}")
            return []
        
        logger.info(f"Loading document: {file_path.name}")
        
        if suffix == '.pdf':
            return self.load_pdf_file(file_path)
        elif suffix == '.csv':
            return self.load_csv_file(file_path)
        elif suffix in {'.txt', '.md', '.html'}:
            return self.load_text_file(file_path)
        else:
            logger.warning(f"No loader available for {suffix}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        """Load all documents from the archive directory"""
        all_documents = []
        
        if not self.archive_dir.exists():
            logger.error(f"Archive directory does not exist: {self.archive_dir}")
            return all_documents
        
        # Get all supported files
        supported_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            supported_files.extend(self.archive_dir.glob(f"*{ext}"))
            supported_files.extend(self.archive_dir.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        supported_files = sorted(set(supported_files))
        
        if not supported_files:
            logger.warning(f"No supported files found in {self.archive_dir}")
            return all_documents
        
        logger.info(f"Found {len(supported_files)} files to load")
        
        # Load each file
        for file_path in supported_files:
            try:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} document(s) from {file_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                continue
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about documents in the archive"""
        stats = {
            'total_files': 0,
            'by_type': {},
            'total_size_mb': 0.0,
            'files': []
        }
        
        if not self.archive_dir.exists():
            return stats
        
        for file_path in self.archive_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                file_type = file_path.suffix.lower()
                size_mb = file_path.stat().st_size / (1024 * 1024)
                
                stats['total_files'] += 1
                stats['total_size_mb'] += size_mb
                
                if file_type not in stats['by_type']:
                    stats['by_type'][file_type] = 0
                stats['by_type'][file_type] += 1
                
                stats['files'].append({
                    'name': file_path.name,
                    'type': file_type,
                    'size_mb': round(size_mb, 2)
                })
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats
        