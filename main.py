"""
RAG-eddy: A simple RAG chatbot for your documents
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

import config
from document_loader import DocumentLoader
from vector_store import VectorStore
from llm_manager import LLMManager
from query_engine import QueryEngine

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGeddy:
    """Simple RAG chatbot with document management"""
    
    def __init__(self):
        self.document_loader = DocumentLoader(config.ARCHIVE_DIR)
        self.vector_store = VectorStore(
            config.VECTOR_STORE_DIR,
            config.EMBEDDING_MODEL,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP
        )
        self.query_engine = None
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        config.ARCHIVE_DIR.mkdir(exist_ok=True)
        config.VECTOR_STORE_DIR.parent.mkdir(exist_ok=True)
        config.MODELS_DIR.mkdir(exist_ok=True)
        logger.info(f"Archive directory: {config.ARCHIVE_DIR}")
    
    def _print_header(self):
        """Improved application header"""
        print("\n" + "="*60)
        print("🤖 RAG-eddy: Document Chat Assistant".center(60))
        print("="*60)
        print(f"📁 Archive: {config.ARCHIVE_DIR}".center(60))
        print(f"💾 Vector Store: {config.VECTOR_STORE_DIR}".center(60))
        print("="*60)
    
    def _print_menu(self):
        """Enhanced menu with clearer options"""
        print("\n📋 MAIN MENU:")
        print("1. 💬 Chat with documents")
        print("2. 🔨 Create/Rebuild vector index")
        print("3. 🔄 Update index (add new documents)")
        print("4. 📄 View document archive")
        print("5. ℹ️  System information")
        print("6. 🚪 Exit")
        print("-"*60)
    
    def run(self):
        self._print_header()
        while True:
            try:
                self._print_menu()
                choice = input(">>> Select option (1-6): ").strip()
                
                if choice == '1': self.chat_interface()
                elif choice == '2': self.create_vector_store()
                elif choice == '3': self.update_vector_store()
                elif choice == '4': self.list_documents()
                elif choice == '5': self.show_system_info()
                elif choice == '6': 
                    print("\n👋 Goodbye!")
                    break
                else:
                    print("⚠️  Invalid choice. Please try again.")
            except Exception as e:
                logger.error(f"Menu error: {e}")
                print(f"❌ System error: {e}")
    
    def chat_interface(self):
        """Interactive chat with documents"""
        print("\n💬 Starting chat interface...")
        
        # Check if vector store exists
        if not self.vector_store.exists():
            print("❌ No vector store found. Please create one first (option 2).")
            input("\nPress Enter to continue...")
            return
        
        # Initialize query engine if not already done
        if not self.query_engine:
            print("🔧 Initializing query engine...")
            self.query_engine = QueryEngine(self.vector_store)
            
            if not self.query_engine.initialize():
                print("❌ Failed to initialize query engine.")
                input("\nPress Enter to continue...")
                return
        
        print("\n" + "="*60)
        print("💬 CHAT MODE - Type '/exit' to return to menu")
        print("💡 Tip: Ask questions about your documents!")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['/exit', '/quit']:
                    break
                
                if not user_input:
                    continue
                
                # Handle query safely
                response = self.query_engine.query(user_input)
                if not response:
                    print("⚠️  No response received. Please try again.")
                    continue
                
                # Print sources
                if response and hasattr(response, 'source_nodes'):
                    sources = self.query_engine.format_sources(response.source_nodes)
                    print(f"\n{sources}")
                
            except KeyboardInterrupt:
                print("\n⚠️  Type '/exit' to return to menu")
            except Exception as e:
                print(f"❌ Error: {e}")
        print("Returning to main menu...")
    
    def create_vector_store(self):
        """Create or rebuild the vector store"""
        print("\n🔨 Creating vector store...")
        
        # Check for existing vector store
        if self.vector_store.exists():
            confirm = input("⚠️  Vector store already exists. Rebuild? (y/n): ").lower()
            if confirm != 'y':
                return
        
        # Load documents
        print("📄 Loading documents from archive...")
        documents = self.document_loader.load_all_documents()
        
        if not documents:
            print("❌ No documents found in archive folder.")
            print(f"📁 Please add documents to: {config.ARCHIVE_DIR}")
            print(f"📋 Supported formats: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS)}")
            input("\nPress Enter to continue...")
            return
        
        print(f"✅ Found {len(documents)} documents")
        
        # Create vector store
        print("🔧 Building vector store...")
        if self.vector_store.create_index(documents, overwrite=True):
            print("✅ Vector store created successfully!")
            # Reset query engine to use new index
            self.query_engine = None
        else:
            print("❌ Failed to create vector store.")
        
        input("\nPress Enter to continue...")
    
    def update_vector_store(self):
        """Update vector store with new documents"""
        print("\n🔄 Updating vector store...")
        
        if not self.vector_store.exists():
            print("❌ No vector store found. Please create one first (option 2).")
            input("\nPress Enter to continue...")
            return
        
        # Get list of indexed documents
        indexed_docs = self.vector_store.get_indexed_documents()
        
        # Get all documents in archive
        all_documents = self.document_loader.load_all_documents()
        
        # Find new documents
        new_documents = []
        for doc in all_documents:
            doc_name = doc.metadata.get('file_name', '')
            if doc_name and doc_name not in indexed_docs:
                new_documents.append(doc)
        
        if not new_documents:
            print("✅ No new documents found. Vector store is up to date.")
            input("\nPress Enter to continue...")
            return
        
        print(f"📄 Found {len(new_documents)} new documents")
        
        # Add new documents to vector store
        print("🔧 Adding new documents to vector store...")
        if self.vector_store.add_documents(new_documents):
            print("✅ Vector store updated successfully!")
            # Reset query engine to use updated index
            self.query_engine = None
        else:
            print("❌ Failed to update vector store.")
        
        input("\nPress Enter to continue...")
    
    def list_documents(self):
        """List all documents in archive"""
        print("\n📄 Documents in archive:")
        print("-"*40)
        
        found_docs = False
        doc_count = 0
        total_size = 0
        
        for file_path in sorted(config.ARCHIVE_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                size = file_path.stat().st_size
                size_mb = size / (1024 * 1024)
                print(f"  📄 {file_path.name} ({size_mb:.2f} MB)")
                found_docs = True
                doc_count += 1
                total_size += size
        
        if not found_docs:
            print("  ❌ No documents found in archive folder")
            print(f"  📁 Add documents to: {config.ARCHIVE_DIR}")
        else:
            total_mb = total_size / (1024 * 1024)
            print("-"*40)
            print(f"  Total: {doc_count} documents ({total_mb:.2f} MB)")
        
        # Check if documents are indexed
        if self.vector_store.exists():
            indexed_count = len(self.vector_store.get_indexed_documents())
            print(f"\n  🔍 Indexed documents: {indexed_count}")
            if indexed_count < doc_count:
                print(f"  ⚠️  {doc_count - indexed_count} documents not indexed")
        else:
            print("\n  ⚠️  No vector store created yet")
        
        input("\nPress Enter to continue...")
    
    def show_system_info(self):
        """Show system information"""
        print("\nℹ️  System Information:")
        print("-"*40)
        print(f"📁 Archive directory: {config.ARCHIVE_DIR}")
        print(f"💾 Vector store: {config.VECTOR_STORE_DIR}")
        print(f"🤖 LLM model: {os.path.basename(config.LLM_MODEL)}")
        print(f"🔤 Embedding model: {config.EMBEDDING_MODEL}")
        print(f"🧵 CPU threads: {config.NUM_THREADS}")
        print(f"🌡️  Temperature: {config.TEMPERATURE}")
        print(f"📏 Chunk size: {config.CHUNK_SIZE}")
        print(f"🔗 Chunk overlap: {config.CHUNK_OVERLAP}")
        
        # Check model status
        model_path = config.MODELS_DIR / os.path.basename(config.LLM_MODEL)
        if model_path.exists():
            size_gb = model_path.stat().st_size / (1024**3)
            print(f"\n✅ LLM model downloaded ({size_gb:.2f} GB)")
        else:
            print(f"\n⚠️  LLM model not downloaded")
        
        # Check vector store status
        if self.vector_store.exists():
            print("✅ Vector store exists")
            doc_count = len(self.vector_store.get_indexed_documents())
            print(f"   Documents indexed: {doc_count}")
        else:
            print("❌ Vector store not created")
        
        input("\nPress Enter to continue...")

def main():
    """Main entry point"""
    try:
        app = RAGeddy()
        app.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
