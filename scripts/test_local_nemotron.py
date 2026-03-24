#!/usr/bin/env python3
import time
import psutil
import mlx.core as mx
from mlx_flash import FlashConfig, FlashGenerationLoop
from pathlib import Path

def get_rss_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def run_test():
    # Local model path from LM Studio
    model_path = "/Users/granite/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit"
    
    if not Path(model_path).exists():
        print(f"[!] Error: Model path not found: {model_path}")
        return

    print(f"[i] Initializing FlashGenerationLoop from: {model_path}")
    print(f"[i] Initial RSS: {get_rss_mb():.0f} MB")
    
    # 3GB RAM Budget - Forces streaming for 30B model
    config = FlashConfig(
        ram_budget_gb=3.0,
        debug=True,
    )
    
    try:
        flash_loop = FlashGenerationLoop(model_path, config=config)
        
        prompt = "Explain the concept of quantum entanglement in simple terms."
        print(f"\nPrompt: {prompt}")
        print("Response: ", end="", flush=True)
        
        start_time = time.time()
        token_count = 0
        
        for chunk in flash_loop.stream_generate(prompt, max_tokens=50, temp=0.7):
            print(chunk, end="", flush=True)
            token_count += 1
            if token_count >= 50:
                break
        
        elapsed = time.time() - start_time
        print(f"\n\n[Stats] Generated {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.2f} t/s)")
        print(f"[Stats] Final Metal Memory: {mx.get_active_memory()/1e6:.0f} MB")
        print(f"[Stats] Final RSS Memory: {get_rss_mb():.0f} MB")
        
    finally:
        if 'flash_loop' in locals():
            flash_loop.shutdown()

if __name__ == "__main__":
    run_test()
