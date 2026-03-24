#!/usr/bin/env python3
"""Diagnostic: pinpoint exactly where Metal OOM occurs for Nemotron-30B."""
import gc
import os
import sys
import psutil
import mlx.core as mx
import mlx_lm

def rss_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def metal_mb():
    try:
        return mx.metal.get_active_memory() / 1e6
    except AttributeError:
        return mx.get_active_memory() / 1e6

def checkpoint(label):
    print(f"  [{label:40s}] RSS={rss_mb():8.1f} MB | Metal={metal_mb():8.1f} MB")
    sys.stdout.flush()

def main():
    model_path = "/Volumes/1tb_ssd/mlx-community/Nemotron-Cascade-2-30B-A3B-4bit"
    
    print("="*80)
    print("NEMOTRON-30B METAL OOM DIAGNOSTIC")
    print("="*80)
    checkpoint("Baseline")
    
    # Step 1: lazy load
    print("\n[1] Loading model with lazy=True...")
    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    checkpoint("After mlx_lm.load(lazy=True)")
    
    # Step 2: find the architecture
    inner = getattr(model, "model", getattr(model, "backbone", model))
    layers = getattr(inner, "layers", getattr(inner, "h", None))
    print(f"\n[2] Model architecture: {type(model).__name__}")
    print(f"    Inner: {type(inner).__name__}")
    print(f"    Num layers: {len(layers)}")
    print(f"    Layer types: {set(type(l).__name__ for l in layers)}")
    
    # Step 3: check hybrid pattern
    hybrid = getattr(model, "hybrid_override_pattern", None) or getattr(inner, "hybrid_override_pattern", None)
    if hybrid is None:
        # check config
        try:
            from mlx_lm.utils import load_config
            cfg = load_config(model_path)
            hybrid = cfg.get("hybrid_override_pattern", "N/A")
        except:
            hybrid = "N/A"
    print(f"    Hybrid pattern: {hybrid}")
    
    # Step 4: measure per-layer weight sizes
    print(f"\n[3] Per-layer weight analysis (first 5 layers):")
    for i in range(min(5, len(layers))):
        layer = layers[i]
        param_count = 0
        for name, param in layer.named_modules():
            pass
        # Count parameters via leaf_modules
        total_bytes = 0
        for p_name, p in layer.parameters().items() if hasattr(layer.parameters(), 'items') else []:
            pass
        block_type = getattr(layer, "block_type", "?")
        print(f"    Layer {i}: type={type(layer).__name__}, block_type={block_type}")
        # Check for MoE specific components
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp
            if hasattr(mlp, 'experts'):
                n_experts = len(mlp.experts) if hasattr(mlp.experts, '__len__') else "?"
                print(f"           MoE: {n_experts} experts")
            if hasattr(mlp, 'shared_experts'):
                print(f"           Has shared expert")
    
    checkpoint("After inspection")
    
    # Step 5: try embedding only
    print(f"\n[4] Testing embedding only...")
    embed = (getattr(inner, "embed_tokens", None) or 
             getattr(inner, "embeddings", None) or
             getattr(inner, "wte", None))
    tokens = mx.array(tokenizer.encode("Hello"))[None]
    print(f"    Tokens shape: {tokens.shape}")
    h = embed(tokens)
    mx.eval(h)
    checkpoint("After embedding + eval")
    
    # Step 6: try a single layer
    print(f"\n[5] Testing single layer forward (layer 0)...")
    layer0 = layers[0]
    block_type_0 = getattr(layer0, "block_type", "?")
    print(f"    Layer 0 type: {type(layer0).__name__}, block_type={block_type_0}")
    
    try:
        import inspect
        sig = inspect.signature(layer0.__call__)
        print(f"    Signature: {sig}")
        
        call_kwargs = {}
        params = sig.parameters
        if "mask" in params:
            call_kwargs["mask"] = None
        if "cache" in params:
            call_kwargs["cache"] = None
        
        output = layer0(h, **call_kwargs)
        if isinstance(output, (list, tuple)):
            h_out = output[0]
        else:
            h_out = output
        mx.eval(h_out)
        checkpoint("After layer 0 + eval")
        
        # Clean up
        
    except Exception as e:
        print(f"    ERROR on layer 0: {type(e).__name__}: {e}")
        checkpoint("After layer 0 error")
    
    # Step 7: try layer 1
    print(f"\n[6] Testing layer 1...")
    try:
        layer1 = layers[1]
        block_type_1 = getattr(layer1, "block_type", "?")
        print(f"    Layer 1 type: {type(layer1).__name__}, block_type={block_type_1}")
        
        sig = inspect.signature(layer1.__call__)
        call_kwargs = {}
        params = sig.parameters
        if "mask" in params:
            call_kwargs["mask"] = None
        if "cache" in params:
            call_kwargs["cache"] = None
        
        output = layer1(h_out, **call_kwargs)
        if isinstance(output, (list, tuple)):
            h_out2 = output[0]
        else:
            h_out2 = output
        mx.eval(h_out2)
        checkpoint("After layer 1 + eval")
    except Exception as e:
        print(f"    ERROR on layer 1: {type(e).__name__}: {e}")
        checkpoint("After layer 1 error")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
