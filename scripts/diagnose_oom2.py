#!/usr/bin/env python3
"""Diagnostic: test if weight reload actually frees Metal memory."""
import gc
import os
import sys
import psutil
import mlx.core as mx
import mlx_lm

def metal_mb():
    try:
        return mx.metal.get_active_memory() / 1e6
    except AttributeError:
        return mx.get_active_memory() / 1e6

def peak_mb():
    try:
        return mx.metal.get_peak_memory() / 1e6
    except AttributeError:
        return mx.get_peak_memory() / 1e6

def checkpoint(label):
    print(f"  [{label:45s}] Metal active={metal_mb():8.1f} MB | peak={peak_mb():8.1f} MB")
    sys.stdout.flush()

def main():
    model_path = "/Volumes/1tb_ssd/mlx-community/Nemotron-Cascade-2-30B-A3B-4bit"
    
    print("="*90)
    print("NEMOTRON-30B: FULL LAYER-BY-LAYER DIAGNOSTIC (first 10 layers)")
    print("="*90)
    
    model, tokenizer = mlx_lm.load(model_path, lazy=True)
    checkpoint("After load")
    
    inner = getattr(model, "model", getattr(model, "backbone", model))
    layers = inner.layers
    embed = getattr(inner, "embed_tokens", None) or getattr(inner, "embeddings", None)
    
    # Hybrid pattern from config
    pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    print(f"Hybrid pattern: {pattern}")
    print(f"Pattern length: {len(pattern)}, Num layers: {len(layers)}")
    
    tokens = mx.array(tokenizer.encode("Hi"))[None]
    h = embed(tokens)
    mx.eval(h)
    checkpoint("After embedding")
    
    # Now iterate through the first 10 layers, doing what FlashLLM does:
    # eval each layer, then reload weights to orphan them, then clear cache
    import re
    import json
    import struct
    from pathlib import Path
    
    model_path_p = Path(model_path)
    layer_regex = re.compile(r'\b(?:layers|h|blocks)\.(\d+)\.')
    local_regex = re.compile(r'\b(?:layers|h|blocks)\.\d+\.(.*)$')
    
    n_layers = len(layers)
    
    # Build weight index for reloading
    sf_files = sorted(model_path_p.glob("*.safetensors"))
    layer_weight_index = [[] for _ in range(n_layers)]
    for sf in sf_files:
        with open(sf, "rb") as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_len).decode('utf-8'))
        layer_keys = {}
        for key in header:
            if key == "__metadata__":
                continue
            m = layer_regex.search(key)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < n_layers:
                    layer_keys.setdefault(idx, []).append(key)
        for idx, keys in layer_keys.items():
            layer_weight_index[idx].append((str(sf), keys))
    
    def reload_layer(idx):
        entries = layer_weight_index[idx]
        if not entries:
            return
        layer = layers[idx]
        fresh_weights = []
        for sf_path, keys in entries:
            lazy_dict = mx.load(sf_path)
            for key in keys:
                if key in lazy_dict:
                    m2 = local_regex.search(key)
                    if m2:
                        fresh_weights.append((m2.group(1), lazy_dict[key]))
        if fresh_weights:
            layer.load_weights(fresh_weights, strict=False)
    
    test_layers = n_layers
    for i in range(test_layers):
        layer = layers[i]
        bt = getattr(layer, "block_type", "?")
        
        # Run layer
        import inspect
        sig = inspect.signature(layer.__call__)
        kw = {}
        if "mask" in sig.parameters:
            kw["mask"] = None
        if "cache" in sig.parameters:
            kw["cache"] = None
        
        output = layer(h, **kw)
        if isinstance(output, (list, tuple)):
            h = output[0]
        else:
            h = output
        mx.eval(h)
        mx.synchronize()
        checkpoint(f"Layer {i:2d} ({bt}) AFTER eval")
        
        # Now reload weights to orphan materialized arrays
        reload_layer(i)
        
        print()
    
    print("="*90)
    print("DIAGNOSTIC COMPLETE")

if __name__ == "__main__":
    main()
