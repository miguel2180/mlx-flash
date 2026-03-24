import time
import os
import mlx.core as mx
import numpy as np
from pathlib import Path

from mlx_flash import FlashConfig
from mlx_flash.engine.engine import FlashEngine
from mlx_flash.safetensors_mmap import SafetensorsMmapCache

# Create a small synthetic model if needed
from benchmarks.run_synthetic_proof import create_massive_synthetic

def measure_generation(engine, prompt, num_tokens=2, label=""):
    print(f"\n--- {label} ---")
    mx.metal.reset_peak_memory()
    
    t0 = time.perf_counter()
    x = prompt
    latencies = []
    
    for i in range(num_tokens):
        t_step0 = time.perf_counter()
        logits = engine(x)
        x = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(x)
        mx.synchronize()
        t_step1 = time.perf_counter()
        latencies.append(t_step1 - t_step0)
        
    t1 = time.perf_counter()
    
    peak_mem = mx.metal.get_peak_memory() / (1024**2)
    active_mem = mx.metal.get_active_memory() / (1024**2)
    cache_mem = mx.metal.get_cache_memory() / (1024**2)
    
    tok_sec = num_tokens / (t1 - t0)
    avg_latency = np.mean(latencies) * 1000
    
    print(f"Speed: {tok_sec:.2f} tokens/sec")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Peak Metal Memory: {peak_mem:.2f} MB")
    print(f"Active Metal Memory: {active_mem:.2f} MB")
    print(f"Cache Metal Memory: {cache_mem:.2f} MB")

if __name__ == "__main__":
    model_dir = Path("/tmp/mlx_synthetic_proof")
    if not (model_dir / "model.safetensors").exists():
        create_massive_synthetic(model_dir, n_layers=16, hidden_dim=2048)
        
    import mlx_lm
    model_lazy, tokenizer = mlx_lm.load(str(model_dir), lazy=True)
    
    prompt = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    
    config = FlashConfig(
        enabled=True, 
        tiled_execution=True, 
        tile_size=512, 
        pipelined_execution=True,
        debug=False
    )
    
    engine_with = FlashEngine(model_lazy, tokenizer, config)
    engine_with.mmap_cache = SafetensorsMmapCache(model_dir)
    
    # warmup
    x = engine_with(prompt)
    mx.eval(x)
    
    
    
    # re-init to ensure clean state
    engine_without = FlashEngine(model_lazy, tokenizer, config)
    engine_without.mmap_cache = SafetensorsMmapCache(model_dir)
    
    # warmup
    x = engine_without(prompt)
    mx.eval(x)
    
