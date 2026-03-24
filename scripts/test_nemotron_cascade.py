#!/usr/bin/env python3
import time
import psutil
import mlx_lm
import mlx.core as mx
from mlx_flash import FlashConfig, FlashGenerationLoop

def get_rss_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def run_test():
    model_path = "/Volumes/1tb_ssd/mlx-community/Nemotron-Cascade-2-30B-A3B-4bit"
    
    print(f"[i] Initializing FlashGenerationLoop from: {model_path}")
    print(f"[i] Initial RSS: {get_rss_mb():.0f} MB")
    
    # 3GB RAM Budget
    config = FlashConfig(
        ram_budget_gb=3.0,
        pipeline_depth=2,
        debug=True,
    )
    
    try:
        flash_loop = FlashGenerationLoop(model_path, config=config)
        
        prompts = [
            "Explain the difference between deep learning and classical machine learning in 3 sentences.",
            "Write a short Python function to calculate the Fibonacci sequence.",
            "What is the capital of France?"
        ]
        
        for idx, prompt in enumerate(prompts):
            print(f"\n--- Test {idx+1} ---")
            print(f"Prompt: {prompt}")
            print("Response: ", end="", flush=True)
            
            start_time = time.time()
            token_count = 0
            
            for chunk in flash_loop.stream_generate(prompt, max_tokens=100, temp=0.0):
                print(chunk, end="", flush=True)
                token_count += 1
            
            elapsed = time.time() - start_time
            print(f"\n\n[Stats] Generated {token_count} tokens in {elapsed:.2f}s ({token_count/elapsed:.2f} t/s)")
            try:
                metal_mb = mx.metal.get_active_memory() / 1e6
            except AttributeError:
                metal_mb = mx.get_active_memory() / 1e6
            print(f"[Stats] Metal Memory: {metal_mb:.0f} MB")
            print(f"[Stats] RSS Memory: {get_rss_mb():.0f} MB")
            
    finally:
        if 'flash_loop' in locals():
            flash_loop.shutdown()

if __name__ == "__main__":
    run_test()
