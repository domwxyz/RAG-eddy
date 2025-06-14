import os
from pathlib import Path
from typing import Optional
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
import requests
import logging
import config

logger = logging.getLogger(__name__)

class LLMManager:
    """Manage the LlamaCPP model"""
    
    def __init__(self):
        self.model_url = config.LLM_MODEL
        self.models_dir = config.MODELS_DIR
        self.temperature = config.TEMPERATURE
        self.num_threads = config.NUM_THREADS
        self.max_tokens = config.MAX_TOKENS
        self.llm = None
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
    
    def initialize(self) -> bool:
        try:
            model_name = os.path.basename(self.model_url)
            local_path = self.models_dir / model_name
            
            # Download with progress if needed
            if not local_path.exists():
                print(f"⏳ Downloading model: {model_name}")
                if not self._download_model(self.model_url, local_path):
                    return False
            else:
                print(f"✅ Using existing model: {model_name}")
            
            # Initialize with better default parameters
            self.llm = LlamaCPP(
                model_path=str(local_path),
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
                context_window=8192,  # Increased for better context
                model_kwargs={
                    "n_threads": max(4, self.num_threads),  # Ensure minimum threads
                    "n_batch": 512,
                    "n_gpu_layers": 0,
                    "verbose": False
                }
            )
            
            # Simple test instead of complex validation
            try:
                test_response = self.llm.complete("Hello")
                if not test_response.text.strip():
                    raise ValueError("Empty test response")
            except Exception as e:
                logger.error(f"Model test failed: {e}")
                return False
            
            Settings.llm = self.llm
            print("✅ LLM ready")
            return True
        except Exception as e:
            logger.exception(f"LLM init failed")
            return False
    
    def _download_model(self, url: str, local_path: Path) -> bool:
        """Download model from URL with progress"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            # Create a temporary file first
            temp_path = local_path.with_suffix('.tmp')
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r⬇️  Downloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
            
            print()  # New line after download
            
            # Move temp file to final location
            temp_path.rename(local_path)
            logger.info(f"Model downloaded successfully: {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def get_llm(self) -> Optional[LlamaCPP]:
        """Get the LLM instance"""
        return self.llm
        