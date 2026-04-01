import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from .config import FlashConfig
from .manager import FlashManager

def main():
    parser = argparse.ArgumentParser(
        description="MLX-Flash: Out-of-Core Weight Streaming for Large Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model & Data
    parser.add_argument("--model", type=str, help="HuggingFace repo ID or local path to model")
    parser.add_argument("--modelfile", type=str, help="Path to an Ollama-style Modelfile")
    parser.add_argument("--prompt", type=str, default="Write a quick poem about an M4 Macbook.", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0.0 for greedy)")
    
    # Memory & Performance
    parser.add_argument("--ram", type=float, help="Target RAM budget in GB for weight residence")
    parser.add_argument("--kv-quant", type=int, choices=[0, 4, 8], help="KV cache quantization bits (0 to disable)")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for memory-bounded execution")
    parser.add_argument("--no-pipeline", action="store_true", help="Disable overlapping IO and compute")
    
    # Diagnostics
    parser.add_argument("--debug", action="store_true", help="Enable verbose diagnostics and profiler reports")
    
    args = parser.parse_args()

    # 1. Build Config
    if args.modelfile:
        from .integration.modelfile import parse_flash_directives
        modelfile_path = Path(args.modelfile)
        if not modelfile_path.exists():
            print(f"[!] Error: Modelfile not found at {args.modelfile}")
            sys.exit(1)
        config = parse_flash_directives(modelfile_path.read_text())
        # Overrides from CLI
        if args.ram: config.ram_budget_gb = args.ram
        if args.kv_quant is not None: 
            config.kv_cache_quantized = (args.kv_quant > 0)
            config.kv_cache_bits = args.kv_quant if args.kv_quant > 0 else 8
        if args.debug: config.debug = True
    else:
        if not args.model:
            print("[!] Error: Either --model or --modelfile must be specified.")
            sys.exit(1)
        config = FlashConfig(
            enabled=True,
            ram_budget_gb=args.ram if args.ram else 4.0,
            pipelined_execution=not args.no_pipeline,
            tiled_execution=True, 
            tile_size=args.tile_size,
            kv_cache_quantized=(args.kv_quant is not None and args.kv_quant > 0),
            kv_cache_bits=args.kv_quant if (args.kv_quant and args.kv_quant > 0) else 8,
            debug=args.debug
        )

    model_path = args.model
    if args.modelfile and not model_path:
        # Try to extract FROM from modelfile if possible
        # This logic could be added to modelfile.py, but for now we'll just check
        with open(args.modelfile, "r") as f:
            for line in f:
                if line.upper().startswith("FROM "):
                    model_path = line.split(None, 1)[1].strip()
                    break
    
    if not model_path:
        print("[!] Error: No model path found (provide --model or FROM in Modelfile).")
        sys.exit(1)

    print(f"[*] Starting Flash Engine...")
    print(f"[*] Model: {model_path}")
    print(f"[*] RAM Budget: {config.ram_budget_gb} GB")
    
    # 2. Orchestrate Load
    manager = FlashManager(config)
    try:
        t0 = time.perf_counter()
        model, tokenizer = manager.load(model_path)
        t1 = time.perf_counter()
        print(f"[*] Model loaded in {t1-t0:.2f}s (Lazy)")
        
        print(f"\nPrompt: {args.prompt}\n" + "-"*40)
        
        # 3. Stream Generate
        tokens_count = 0
        gen_t0 = time.perf_counter()
        
        print("Generation:", end=" ", flush=True)
        
        for segment in model.stream_generate(
            args.prompt, 
            max_tokens=args.max_tokens, 
            temp=args.temp
        ):
            tokens_count += 1
            if segment:
                print(segment, end="", flush=True)
            elif args.debug:
                # In debug mode, show a dot for segments that don't yield text
                print(".", end="", flush=True)
            
        gen_t1 = time.perf_counter()
        duration = gen_t1 - gen_t0
        
        print("\n" + "-"*40)
        print(f"[*] Done. Generated {tokens_count} tokens in {duration:.2f}s ({tokens_count/duration:.2f} tok/s)")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user.")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        manager.shutdown()

if __name__ == "__main__":
    main()
