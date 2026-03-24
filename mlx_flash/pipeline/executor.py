import mlx.core as mx
import mlx.nn as nn

class PipelinedExecutor:
    """
    Executes a transformer layer by breaking it into sub-components and explicitly 
    interleaving I/O prefetch requests with GPU compute dispatches.
    
    This replaces the monolithic `layer.forward(x)` with a state machine that
    understands the sub-tensors (wqkv, wo, w_gate_up, w_down).
    """
    def __init__(self, mmap_cache):
        self.mmap_cache = mmap_cache
        
    def _enqueue_tensor(self, layer_idx: int, tensor_name_hint: str):
        """Asks the background worker to load specific tensors based on name hints."""
        if not self.mmap_cache:
            return
            
        ranges = self.mmap_cache.get_layer_ranges(layer_idx)
        # In a fully realized system, we would filter `ranges` by tensor_name_hint
        # For now, we use the existing layer-level prefetch as a coarse fallback
        # if fine-grained isn't implemented in the cache yet.
        for _, (start, end, filename, dtype) in ranges.items():
            align_bytes = 1
            if dtype == "q4_0": align_bytes = 18
            elif dtype == "q8_0": align_bytes = 34
            elif dtype.startswith("q"): align_bytes = 256
            self.mmap_cache.prefetch_worker.enqueue(filename, start, end - start, layer_idx, align_bytes=align_bytes)

    def _wait_for_layer(self, layer_idx: int):
        if self.mmap_cache and hasattr(self.mmap_cache, 'wait_for_layer'):
            self.mmap_cache.wait_for_layer(layer_idx)

    def execute_moe_layer(self, x: mx.array, layer: nn.Module, layer_idx: int, moe_prefetcher, mask=None, cache=None) -> mx.array:
        """
        Executes a Mixture of Experts layer, heavily overlapping the router computation 
        with speculative and deterministic expert prefetching.
        """
        # ==========================================
        # PHASE 1: PRE-ATTENTION PIPELINE
        # ==========================================
        self._enqueue_tensor(layer_idx, "attention")
        
        if hasattr(layer, "input_layernorm"): h = layer.input_layernorm(x)
        elif hasattr(layer, "norm"): h = layer.norm(x)
        else: h = x
            
        # ==========================================
        # PHASE 2: ATTENTION OVERLAP
        # ==========================================
        self._wait_for_layer(layer_idx) 
        
        # IO: Instead of fetching the whole MLP, just fetch the tiny Router
        self._enqueue_tensor(layer_idx, "router")
        
        attn_module = getattr(layer, "self_attn", getattr(layer, "attention", None))
        if attn_module is not None:
            call_kwargs = {}
            if mask is not None: call_kwargs["mask"] = mask
            if cache is not None: call_kwargs["cache"] = cache
            
            attn_out = attn_module(h, **call_kwargs)
            attn_out = attn_out[0] if isinstance(attn_out, (list, tuple)) else attn_out
            x = x + attn_out
            mx.eval(x)
            mx.metal.clear_cache()
            
        # ==========================================
        # PHASE 3: MoE ROUTING & SPECULATIVE FETCH
        # ==========================================
        if hasattr(layer, "post_attention_layernorm"): h = layer.post_attention_layernorm(x)
        elif hasattr(layer, "norm_f"): h = layer.norm_f(x)
        else: h = x
            
        self._wait_for_layer(layer_idx) # Wait for router
        
        mlp_module = getattr(layer, "mlp", getattr(layer, "mixer", None))
        
        if mlp_module is not None and hasattr(mlp_module, "gate"):
            seq_len = h.shape[1] if len(h.shape) > 1 else 1
            
            # 1. GPU: Compute Router Logits
            router = mlp_module.gate
            if hasattr(router, "weight"):
                logits = mx.matmul(h, router.weight.T)
                if hasattr(router, "bias") and router.bias is not None:
                    logits = logits + router.bias
            else:
                logits = router(h)
                
            mx.eval(logits)
            
            top_k = getattr(mlp_module, "num_experts_per_tok", getattr(mlp_module.config, "num_experts_per_tok", 2))
            
            if seq_len == 1:
                # ========================================================
                # FAST PATH: Single Token Decode (Streaming)
                # ========================================================
                import numpy as np
                np_logits = np.array(logits) 
                flat_logits = np_logits.flatten()
                
                if len(flat_logits) > top_k:
                    top_indices = np.argpartition(flat_logits, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(flat_logits[top_indices])[::-1]]
                else:
                    top_indices = np.arange(len(flat_logits))
                    
                active_experts = top_indices.tolist()
                moe_prefetcher.update_history(layer_idx, active_experts)
                
                for exp_idx in active_experts:
                     if moe_prefetcher.cache.get(layer_idx, exp_idx) is None:
                         self._enqueue_tensor(layer_idx, f"experts.{exp_idx}")
                         
                self._wait_for_layer(layer_idx) 
                mlp_out = mlp_module(h)
                x = x + mlp_out
                
            else:
                # ========================================================
                # BATCHED PATH: Expert Pre-Sorting (Prefill Phase)
                # ========================================================
                # Compute routing softmax
                scores = mx.softmax(logits, axis=-1)
                
                # Get top-k indices and values using MLX built-ins
                # MLX topk is available in newer versions, assuming it exists or fallback
                # For safety across MLX versions, we'll use a sort if topk is missing
                if hasattr(mx, "topk"):
                    top_scores, top_indices = mx.topk(scores, k=top_k, axis=-1)
                else:
                    sorted_indices = mx.argsort(scores, axis=-1)
                    top_indices = sorted_indices[..., -top_k:]
                    # Reverse to get descending order
                    top_indices = top_indices[..., ::-1]
                    # Gather scores
                    top_scores = mx.take_along_axis(scores, top_indices, axis=-1)
                
                # Flatten everything to 1D
                flat_indices = mx.reshape(top_indices, (-1,))
                flat_scores = mx.reshape(top_scores, (-1,))
                
                # Normalize scores if required by architecture
                flat_scores = flat_scores / mx.sum(top_scores, axis=-1, keepdims=True).reshape(-1,)
                
                final_mlp_out = mx.zeros_like(h)
                
                # Find unique experts required for this batch
                import numpy as np
                unique_experts = np.unique(np.array(flat_indices)).tolist()
                
                # Dispatch prefetch for all needed experts
                for exp_idx in unique_experts:
                    if moe_prefetcher.cache.get(layer_idx, exp_idx) is None:
                        self._enqueue_tensor(layer_idx, f"experts.{exp_idx}")
                        
                # Execute expert by expert
                for exp_idx in unique_experts:
                    self._wait_for_layer(layer_idx) # Wait for this specific expert's IO
                    
                    # 1. Find which tokens requested this expert
                    token_mask = (flat_indices == exp_idx)
                    
                    if not mx.any(token_mask).item():
                        continue
                        
                    # 2. Extract weights
                    expert_weights = flat_scores[token_mask]
                    
                    # 3. Map flattened index back to original sequence token ID
                    token_ids = mx.arange(seq_len * top_k)[token_mask] // top_k
                    
                    # Ensure dimensions match for extraction
                    # h is [batch, seq, dim], assume batch=1 for standard prefill
                    expert_input = h[0, token_ids, :]
                    
                    # 4. Compute Expert (only for active tokens)
                    expert_module = getattr(mlp_module, "experts")[exp_idx]
                    expert_out = expert_module(expert_input)
                    
                    expert_out = expert_out * mx.expand_dims(expert_weights, axis=-1)
                    
                    # 5. Scatter Add results
                    # MLX scatter requires updates to match indices
                    final_mlp_out[0] = mx.scatter_add(final_mlp_out[0], token_ids, expert_out)
                    
                    mx.eval(final_mlp_out)
                    
                    # Yield VRAM immediately
                    # In a real system we'd explicitly clear this expert if we are over budget
                    mx.metal.clear_cache()
                    
                x = x + final_mlp_out
            
        mx.eval(x)
        mx.metal.clear_cache()
        return x
    def execute_dense_layer(self, x: mx.array, layer: nn.Module, layer_idx: int, mask=None, cache=None) -> mx.array:
        # ==========================================
        # PHASE 1: PRE-ATTENTION PIPELINE
        # ==========================================
        # IO: Start fetching Attention weights
        self._enqueue_tensor(layer_idx, "attention")
        
        # GPU: Input Norm (fast, requires no SSD weights)
        if hasattr(layer, "input_layernorm"):
            h = layer.input_layernorm(x)
        elif hasattr(layer, "norm"):
            h = layer.norm(x)
        else:
            h = x
            
        # ==========================================
        # PHASE 2: ATTENTION OVERLAP
        # ==========================================
        # Wait for Attention weights
        self._wait_for_layer(layer_idx) 
        
        # IO: The moment attention begins computing, prefetch the MLP
        self._enqueue_tensor(layer_idx, "mlp")
        
        # GPU: Attention computation
        # (Assuming the layer has a standard self_attn or attention attribute)
        attn_module = getattr(layer, "self_attn", getattr(layer, "attention", None))
        
        if attn_module is not None:
            call_kwargs = {}
            if mask is not None: call_kwargs["mask"] = mask
            if cache is not None: call_kwargs["cache"] = cache
            
            attn_out = attn_module(h, **call_kwargs)
            # Handle tuple returns (some attention modules return cache states)
            attn_out = attn_out[0] if isinstance(attn_out, (list, tuple)) else attn_out
            
            x = x + attn_out
            
            # Force evaluation and memory clear before moving to MLP
            mx.eval(x)
            if cache is not None:
                 if hasattr(cache, "state") and cache.state is not None:
                     mx.eval(*[s for s in cache.state if s is not None])
                 elif hasattr(cache, "keys") and cache.keys is not None:
                     mx.eval(cache.keys, cache.values)
            mx.metal.clear_cache()
            
        # ==========================================
        # PHASE 3: MLP EXECUTION
        # ==========================================
        # GPU: Post-attention norm
        if hasattr(layer, "post_attention_layernorm"):
            h = layer.post_attention_layernorm(x)
        elif hasattr(layer, "norm_f"): # Specific to some architectures
             h = layer.norm_f(x)
        else:
            h = x
            
        # Wait for MLP weights (likely already loaded due to prefetch in Phase 2)
        self._wait_for_layer(layer_idx)
        
        # GPU: MLP computation
        mlp_module = getattr(layer, "mlp", getattr(layer, "mixer", None))
        if mlp_module is not None:
            mlp_out = mlp_module(h)
            x = x + mlp_out
            
        # Final layer sync
        mx.eval(x)
        mx.metal.clear_cache()
        
        return x
