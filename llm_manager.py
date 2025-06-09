import os
from pathlib import Path
from typing import Optional
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
import requests
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    """Manage the LlamaCPP model"""
    
    def __init__(self, model_url: str, models_dir: Path, temperature: float, num_threads: int, max_tokens: int):
        self.model_url = model_url
        self.models_dir = models_dir
        self.temperature = temperature
        self.num_threads = num_threads
        self.max_tokens = max_tokens
        self.llm = None
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
    
    def initialize(self) -> bool:
        """Initialize the LLM"""
        try:
            # Get local model path
            model_name = os.path.basename(self.model_url)
            local_model_path = self.models_dir / model_name
            
            # Download model if not exists
            if not local_model_path.exists():
                logger.info(f"Downloading model: {model_name}")
                self._download_model(self.model_url, local_model_path)
            
            # Initialize LlamaCPP
            logger.info(f"Loading model: {model_name}")
            self.llm = LlamaCPP(
                model_path=str(local_model_path),
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                context_window=4096,
                model_kwargs={
                    "n_threads": self.num_threads,
                    "n_batch": 512,
                    "use_mlock": True
                },
                verbose=False
            )
            
            # Set as default LLM
            Settings.llm = self.llm
            
            logger.info("LLM initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False
    
    def _download_model(self, url: str, local_path: Path):
        """Download model from URL"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rDownloading: {percent:.1f}%", end='')
        
        print()  # New line after download
    
    def get_llm(self) -> Optional[LlamaCPP]:
        """Get the LLM instance"""
        return self.llm
        