from typing import List, Optional
from llama_index.core import PromptTemplate, Settings
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore
import logging

from llm_manager import LLMManager
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class QueryEngine:
    """Handle queries against the vector store"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.query_engine = None
        self.llm = None
    
    def initialize(self) -> bool:
        """Initialize the query engine"""
        try:
            # Initialize LLM if not already done
            if not self.llm:
                logger.info("Initializing LLM...")
                llm_manager = LLMManager()
                if not llm_manager.initialize():
                    logger.error("Failed to initialize LLM")
                    return False
                self.llm = llm_manager.get_llm()
                Settings.llm = self.llm
            
            # Get vector index
            index = self.vector_store.get_index()
            if not index:
                logger.error("No vector index available")
                return False
            
            # Create custom prompt template
            qa_prompt = PromptTemplate(
                "You are a helpful assistant that answers questions based on the provided documents.\n"
                "Be concise and accurate in your responses.\n\n"
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information, answer the question: {query_str}\n"
                "If you cannot find the answer in the provided context, say 'I don't have enough information in the documents to answer that question.'\n"
            )
            
            # Create query engine with streaming
            self.query_engine = index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=0.3)
                ],
                text_qa_template=qa_prompt,
                streaming=True,
                verbose=False
            )
            
            logger.info("Query engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            return False
    
    def query(self, query_text: str):
        if not self.query_engine:
            raise ValueError("Query engine not initialized")
        
        try:
            response = self.query_engine.query(query_text)
            full_response = ""
            
            # Handle streaming safely
            if hasattr(response, 'response_gen'):
                print("\nRAG-eddy: ", end='', flush=True)
                for text in response.response_gen:
                    print(text, end='', flush=True)
                    full_response += text
                print()  # Ensure new line after response
            else:
                full_response = str(response)
                print(f"\nRAG-eddy: {full_response}")
            
            # Preserve response for source formatting
            response.response = full_response
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}")
            return None  # Return None instead of crashing
    
    def format_sources(self, source_nodes: List[NodeWithScore]) -> str:
        """Format source documents in a clean way"""
        if not source_nodes:
            return ""
        
        output = ["\n📚 Sources:"]
        seen_files = set()
        
        for i, node in enumerate(source_nodes[:3]):  # Limit to top 3 sources
            try:
                # Get metadata
                metadata = node.metadata
                file_name = metadata.get('file_name', 'Unknown')
                
                # Skip if we've already shown this file
                if file_name in seen_files:
                    continue
                seen_files.add(file_name)
                
                # Format source info
                output.append(f"\n  📄 {file_name}")
                
                # Get a snippet of text
                text = node.text[:200] + "..." if len(node.text) > 200 else node.text
                # Clean up the text
                text = text.replace('\n', ' ').strip()
                output.append(f"     \"{text}\"")
                
            except Exception as e:
                logger.error(f"Error formatting source: {e}")
                continue
        
        return "\n".join(output) if len(output) > 1 else ""
