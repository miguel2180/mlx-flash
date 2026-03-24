#!/usr/bin/env python3
import time
import json
import csv
import sys
import os
import psutil
import gc
from pathlib import Path
from dataclasses import asdict
import mlx.core as mx
from mlx_flash import FlashConfig, FlashGenerationLoop

def get_stats():
    gc.collect()
    try:
        metal = mx.get_active_memory() / 1e6
    except:
        metal = 0
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e6
    return {"metal_mb": metal, "rss_mb": rss}

def run_experiment(model_path, config_dict, prompt="Explain quantum physics.", max_tokens=20):
    config = FlashConfig(**config_dict)
    
    start_rss = get_stats()["rss_mb"]
    
    try:
        t0 = time.time()
        loop = FlashGenerationLoop(model_path, config=config)
        init_time = time.time() - t0
        
        peak_metal = 0
        peak_rss = 0
        
        t1 = time.time()
        tokens = 0
        for _ in loop.stream_generate(prompt, max_tokens=max_tokens):
            stats = get_stats()
            peak_metal = max(peak_metal, stats["metal_mb"])
            peak_rss = max(peak_rss, stats["rss_mb"])
            tokens += 1
        
        gen_time = time.time() - t1
        t_ps = tokens / gen_time if gen_time > 0 else 0
        
        loop.shutdown()
        
        return {
            "status": "success",
            "init_time": init_time,
            "tokens_per_sec": t_ps,
            "peak_metal_mb": peak_metal,
            "peak_rss_mb": peak_rss,
            "rss_overhead_mb": peak_rss - start_rss
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    # Configuration Matrix
    model_paths = [
        "/Users/granite/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit"
    ]
    
    budgets = [2.0, 4.0, 8.0]
    chunk_sizes = [32, 128, 512]
    
    results = []
    
    print(f"{'Model':<20} | {'RAM':<5} | {'Chunk':<5} | {'t/s':<6} | {'Peak Metal':<10}")
    print("-" * 65)
    
    for mpath in model_paths:
        mname = Path(mpath).name[:20]
        for budget in budgets:
            for chunk in chunk_sizes:
                cfg = {
                    "ram_budget_gb": budget,
                    "prefill_chunk_size": chunk,
                }
                
                res = run_experiment(mpath, cfg)
                
                if res["status"] == "success":
                    print(f"{mname:<20} | {budget:<5.1f} | {chunk:<5} | {res['tokens_per_sec']:<6.2f} | {res['peak_metal_mb']:<10.0f}")
                    results.append({**cfg, **res, "model": mname})
                else:
                    print(f"{mname:<20} | {budget:<5.1f} | {chunk:<5} | ERROR: {res['error'][:20]}")

    # Save results
    with open("experiment_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[+] Experiments complete. Results saved to experiment_matrix.json")

if __name__ == "__main__":
    main()
