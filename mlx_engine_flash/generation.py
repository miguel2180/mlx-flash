import mlx.core as mx
import mlx.nn as nn
from typing import Any, Optional, Generator, Dict, List, Tuple
from pathlib import Path
import mlx_lm
from .config import FlashConfig

class FlashLLM(nn.Module):
    """
    Wrapper for MLX models that forces synchronous per-layer evaluation.
    This bypasses the 'lazy graph accumulation' problem that causes OOM.
    """
    def __init__(self, base_model: nn.Module, config: FlashConfig):
        super().__init__()
        self.model = base_model
        self.config = config

    def __call__(
        self,
        x: mx.array,
        cache: Optional[list[Any]] = None,
        **kwargs
    ) -> mx.array:
        # Access the backbone (most mlx_lm models wrap the actual transformer in .model)
        backbone = getattr(self.model, "model", self.model)
        
        # 1. Embedding
        x = backbone.embed_tokens(x)
        
        # 2. Sequential Layers
        mask = kwargs.get("mask")
        num_layers = len(backbone.layers)
        
        from .page_cache import prefetch_array
        
        for i, layer in enumerate(backbone.layers):
            # Prefetch the NEXT layer while THIS one is still logic (lazy)
            if i + 1 < num_layers:
                next_layer = backbone.layers[i + 1]
                # Each parameter is an mx.array
                for p in next_layer.parameters().values():
                    if isinstance(p, mx.array):
                        prefetch_array(p)
                    elif isinstance(p, dict): # handle nested params (MoE)
                        for sp in p.values():
                            if isinstance(sp, mx.array):
                                prefetch_array(sp)
            
            l_cache = cache[i] if cache else None
            
            # Forward pass for one layer
            x = layer(x, mask=mask, cache=l_cache)
            
            # Force materialisation of this layer's output NOW
            mx.eval(x)
            
            # If we have a cache, evaluate it too to ensure keys/values are computed
            if l_cache is not None:
                # Modern MLX-LM cache objects have .keys and .values arrays
                mx.eval(l_cache.keys, l_cache.values)
                
                # Disk KV Cache offloading (Advanced)
                if hasattr(self, "disk_cache") and self.disk_cache:
                    token_count = l_cache.keys.shape[1]
                    if token_count > self.config.max_in_memory_kv_tokens:
                        # Evict first half to disk
                        mid = token_count // 2
                        keys_to_evict = l_cache.keys[:, :mid, :]
                        values_to_evict = l_cache.values[:, :mid, :]
                        self.disk_cache.evict_to_disk(i, keys_to_evict, values_to_evict, (0, mid))
                        
                        # Note: In a production version, we would then slice l_cache
                        # but KVCache objects in MLX-LM are often immutable or 
                        # have internal state that's hard to slice externally.
                        # For now, we record the eviction for diagnostics.
                        if self.config.debug:
                            print(f"[flash] Layer {i} evicted {mid} tokens to SSD")

            # Clear Metal pool memory to release weight buffers for this layer
            mx.clear_cache()
            
            if self.config.debug and i % 10 == 0:
                print(f"[flash] Layer {i} evaluated. Metal active: {mx.get_active_memory()/1e6:.1f} MB")

    # 3. Final Norm + Head
        x = backbone.norm(x)
        return self.model.lm_head(x) if hasattr(self.model, "lm_head") else backbone.lm_head(x)

class DiskKVCache:
    """KV cache that evicts old entries to mmap'd files on SSD."""
    
    def __init__(self, n_layers: int, max_in_memory_tokens: int, cache_dir: str):
        self._dir = Path(cache_dir)
        self._dir.mkdir(exist_ok=True, parents=True)
        self._max_in_mem = max_in_memory_tokens
        self._n_layers = n_layers
        self._disk_keys = {}   # layer -> list of paths
        self._disk_values = {}
    
    def evict_to_disk(self, layer_idx: int, keys: mx.array, values: mx.array, token_range: Tuple[int, int]):
        """Evict a range of KV entries to disk (safetensors format)."""
        path = self._dir / f"layer_{layer_idx}_tokens_{token_range[0]}_{token_range[1]}.safetensors"
        mx.save_safetensors(str(path), {"k": keys, "v": values})
        return path
    
    def load_from_disk(self, layer_idx: int, token_range: Tuple[int, int]):
        """Load KV entries back from disk (lazy, mmap-backed)."""
        path = self._dir / f"layer_{layer_idx}_tokens_{token_range[0]}_{token_range[1]}.safetensors"
        data = mx.load(str(path))  # lazy mmap, no Metal allocation yet
        return data["k"], data["v"]

class FlashGenerationLoop:
    """
    High-level generator for flashweight streaming using the FlashLLM wrapper.
    """
    def __init__(self, model_path: str, config: FlashConfig):
        self.config = config
        # Native MLX-LM load with luxury lazy=True
        self.model, self.tokenizer = mlx_lm.load(model_path, {"lazy": True})
        self.flash_model = FlashLLM(self.model, config)
        
        n_layers = len(getattr(self.model, "model", self.model).layers)
        
        # Initialize Cache
        if config.max_kv_size is not None:
            from mlx_lm.models.cache import RotatingKVCache
            self._cache = [
                RotatingKVCache(max_size=config.max_kv_size, keep=config.kv_keep)
                for _ in range(n_layers)
            ]
        else:
            from mlx_lm.models.cache import make_prompt_cache
            self._cache = make_prompt_cache(self.model)
            
        self.disk_cache = None
        if config.kv_cache_dir:
            self.disk_cache = DiskKVCache(
                n_layers, 
                config.max_in_memory_kv_tokens, 
                config.kv_cache_dir
            )
            self.flash_model.disk_cache = self.disk_cache

    def _chunked_prefill(self, prompt_tokens: List[int], chunk_size: int = 512, **kwargs):
        """Process a long prompt in chunks to avoid attention OOM."""
        num_tokens = len(prompt_tokens)
        if chunk_size <= 0 or num_tokens <= chunk_size:
            input_tokens = mx.array(prompt_tokens)[None]
            return self.flash_model(input_tokens, cache=self._cache, **kwargs)

        for i in range(0, num_tokens, chunk_size):
            chunk = prompt_tokens[i : i + chunk_size]
            chunk_input = mx.array(chunk)[None]
            logits = self.flash_model(chunk_input, cache=self._cache, **kwargs)
            mx.eval(logits)
            mx.clear_cache()
        return logits

    def stream_generate(self, prompt: str, max_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        tokens = self.tokenizer.encode(prompt)
        return self.stream_generate_from_tokens(tokens, max_new_tokens=max_tokens, **kwargs)

    def stream_generate_from_tokens(self, tokens: List[int], max_new_tokens: int = 100, **kwargs) -> Generator[str, None, None]:
        # Prefill (possibly chunked)
        logits = self._chunked_prefill(tokens, chunk_size=self.config.prefill_chunk_size, **kwargs)
        
        count = 0
        # Take last logit for next token
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        
        while count < max_new_tokens:
            token_id = next_token.tolist()[0]
            yield self.tokenizer.decode([token_id])
            
            input_tokens = next_token[None]
            logits = self.flash_model(input_tokens, cache=self._cache, **kwargs)
            next_token = mx.argmax(logits[:, -1, :], axis=-1)
            
            count += 1
            if token_id == self.tokenizer.eos_token_id:
                break

    def shutdown(self):
        # Nothing to manually close; OS handles mmap cleanup
        pass
