import sys
import gc
import psutil
import os
import mlx.core as mx
from mlx_flash import FlashConfig, FlashGenerationLoop

def get_rss_mb() -> float:
    gc.collect()
    mx.synchronize()
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def main():
    model_path = "/Volumes/1tb_ssd/mlx-community/Nemotron-Cascade-2-30B-A3B-4bit"
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return

    # Test with 1GB RAM budget to aggressively free Metal memory on 16GB machine
    config = FlashConfig(enabled=True, ram_budget_gb=1.0, debug=True)
    print(f"Loading model {model_path} with 2GB budget...")
    
    rss_start = get_rss_mb()
    loop = FlashGenerationLoop(model_path, config)
    rss_loaded = get_rss_mb()
    print(f"\nModel Loaded. RAM delta: {rss_loaded - rss_start:.1f} MB\n")

    prompts = [
        "What is the capital of France?",
        "Write a haiku about memory leaks.",
        "Explain quantum computing in one sentence."
    ]

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}: {prompt}")
        rss_before = get_rss_mb()
        
        print("Response: ", end="", flush=True)
        # Using mlx_lm sample utils through FlashGenerationLoop
        for text in loop.stream_generate(prompt, max_tokens=50, temp=0.0):
            print(text, end="", flush=True)
        print()
        
        rss_after = get_rss_mb()
        print(f"\n[RAM INFO] Before: {rss_before:.1f} MB | After: {rss_after:.1f} MB | Delta: {rss_after - rss_before:.1f} MB")

if __name__ == "__main__":
    main()
