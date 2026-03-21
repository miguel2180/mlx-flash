import mlx_lm
import mlx.core as mx
from pathlib import Path
from typing import Any, Tuple
from .config import FlashConfig
from .generation import FlashLLM

class FlashManager:
    """
    Orchestrates the Flash Weight Streaming environment.
    """
    def __init__(self, config: FlashConfig = None):
        self.config = config or FlashConfig()
        self.model = None
        self.tokenizer = None

    def load(self, model_path: str | Path) -> Tuple[FlashLLM, Any]:
        """
        Load a model in lazy mode and wrap it for Flash execution.
        """
        path = Path(model_path)
        
        # Avoid recursion if mlx_lm is monkey-patched
        try:
            from .integration.lmstudio import _ORIGINAL_LOAD
            loader = _ORIGINAL_LOAD or mlx_lm.load
        except ImportError:
            loader = mlx_lm.load
            
        # 1. Native lazy load
        model, self.tokenizer = loader(str(path), {"lazy": True})
        
        # 2. Wrap in Flash execution engine
        self.model = FlashLLM(model, self.config)
        
        return self.model, self.tokenizer

    def shutdown(self):
        pass

__all__ = ["FlashManager"]
