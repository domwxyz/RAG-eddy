import logging
import sys
from pathlib import Path
from llama_index.core import PromptTemplate

import config
from document_loader import DocumentLoader
from vector_store import VectorStore
from llm_manager import LLMManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGeddy:
    """Simple RAG chatbot"""
    
    def __init__(self):
        self.document_loader = DocumentLoader(config.ARCHIVE_DIR)
        self.vector_store = VectorStore(
            config.VECTOR_STORE_DIR,
            config.EMBEDDING_MODEL,
            config.CHUNK_SIZE,
            config.CHUNK_OVERLAP
        )
        self.llm_manager = LLMManager(
            config.LLM_MODEL,
            config.MODELS_DIR,
            config.TEMPERATURE,
            config.NUM_THREADS,
            config.MAX_TOKENS
        )
        self.query_engine = None
    
    def setup(self, rebuild_index: bool = False):
        """Set up the RAG system"""
        # Initialize LLM
        logger.info("Initializing LLM...")
        if not self.llm_manager.initialize():
            logger.error("Failed to initialize LLM")
            return False
        
        # Check if we need to build index
        need_to_build = rebuild_index or not config.VECTOR_STORE_DIR.exists()
        
        if not need_to_build:
            # Try to load existing index
            logger.info("Loading existing vector store...")
            if not self.vector_store.load_index():
                logger.info("Failed to load existing vector store, will create new one")
                need_to_build = True
        
        if need_to_build:
            logger.info("Building vector store...")
            documents = self.document_loader.load_all_documents()
            if not documents:
                logger.error("No documents found in archive folder")
                logger.info(f"Please add documents to: {config.ARCHIVE_DIR}")
                logger.info(f"Supported formats: {', '.join(DocumentLoader.SUPPORTED_EXTENSIONS)}")
                return False
            
            if not self.vector_store.create_index(documents, overwrite=True):
                logger.error("Failed to create vector store")
                return False
        
        # Create query engine
        index = self.vector_store.get_index()
        if not index:
            logger.error("No index available")
            return False
        
        # Create custom prompt template
        qa_prompt = PromptTemplate(
            f"{config.SYSTEM_PROMPT}\n\n"
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information, answer the question: {query_str}\n"
            "If the answer is not in the context, say 'I don't have enough information to answer that.'\n"
        )
        
        self.query_engine = index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=qa_prompt,
            streaming=True
        )
        
        logger.info("RAGeddy is ready!")
        return True
    
    def chat(self):
        """Interactive chat interface"""
        if not self.query_engine:
            logger.error("System not initialized. Run setup() first.")
            return
        
        print("\n" + "="*50)
        print("Welcome to RAGeddy! Your document chat assistant.")
        print("Type 'exit' to quit, 'help' for commands")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Handle commands
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'list':
                    self._list_documents()
                    continue
                elif not user_input:
                    continue
                
                # Query the system
                print("\nRAGeddy: ", end='', flush=True)
                response = self.query_engine.query(user_input)
                
                # Stream the response
                for text in response.response_gen:
                    print(text, end='', flush=True)
                print("\n")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"\nError: {e}\n")
    
    def _show_help(self):
        """Show help information"""
        print("\nCommands:")
        print("  exit  - Quit the chat")
        print("  help  - Show this help message")
        print("  list  - List indexed documents")
        print("\n")
    
    def _list_documents(self):
        """List documents in the archive"""
        print("\nIndexed documents:")
        found_docs = False
        for file_path in sorted(config.ARCHIVE_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS:
                print(f"  - {file_path.name}")
                found_docs = True
        if not found_docs:
            print("  No documents found in archive folder")
        print()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAGeddy - Simple Document RAG Chatbot")
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vector index')
    args = parser.parse_args()
    
    # Create and setup RAGeddy
    rag = RAGeddy()
    
    if not rag.setup(rebuild_index=args.rebuild):
        logger.error("Failed to initialize RAGeddy")
        sys.exit(1)
    
    # Start chat interface
    rag.chat()

if __name__ == "__main__":
    main()
