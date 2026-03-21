import time

import mlx_lm
import psutil

from mlx_flash import FlashConfig
from mlx_flash.integration.lmstudio import apply_flash_patch

# 1. SETUP
MODEL_PATH = "/Users/granite/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit"

# 2. CHOOSE YOUR MODE
# - 1.5: Safe Flash Mode (Lowest RAM pressure, works while multi-tasking)
# - 8.0: Performance (Requires 16GB+ free RAM)
RAM_BUDGET = 1.5 

cfg = FlashConfig(
    enabled=True,
    ram_budget_gb=RAM_BUDGET,
    debug=True
)

def get_rss_gb():
    return psutil.Process().memory_info().rss / 1024**3

def run_demo(model, tokenizer, prompt):
    print(f"\nPrompt: {prompt}")
    print(f"Starting RSS: {get_rss_gb():.2f} GB")
    
    t0 = time.monotonic()
    tokens = 0
    
    for response in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=50):
        print(response.text, end="", flush=True)
        tokens += 1
        
    t1 = time.monotonic()
    duration = t1 - t0
    print("\n\n- Stats -")
    print(f"Final RSS: {get_rss_gb():.2f} GB")
    print(f"Speed: {tokens/duration:.2f} tok/s")
    print(f"Time: {duration:.1f}s for {tokens} tokens")

def main():
    print("⚡ MLX-FLASH 30B DEMO ⚡")
    print(f"RAM Budget: {RAM_BUDGET} GB")
    
    apply_flash_patch(cfg)
    
    print("Loading 30B Model (Lazy)...")
    model, tokenizer = mlx_lm.load(MODEL_PATH)
    
    prompts = [
        "In one sentence, explain quantum entanglement.",
        "Write a 3-step recipe for space-faring pasta.",
    ]
    
    for p in prompts:
        run_demo(model, tokenizer, p)
        print("-" * 30)

if __name__ == "__main__":
    main()
