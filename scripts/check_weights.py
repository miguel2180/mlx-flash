#!/usr/bin/env python3
import time
import mlx_lm
from mlx_flash import FlashConfig, FlashGenerationLoop

model_path = "/Volumes/1tb_ssd/mlx-community/Nemotron-Cascade-2-30B-A3B-4bit"
config = FlashConfig(ram_budget_gb=3.0, debug=True)
loop = FlashGenerationLoop(model_path, config=config)

index = loop.flash_model._layer_weight_index
print(f"Index length: {len(index)}")
if len(index) > 0:
    for i in range(2):
        print(f"Layer {i}:")
        for sf_path, keys in index[i]:
            print(f"  {sf_path}: {len(keys)} keys")
            if keys:
                print(f"    e.g. {keys[0]}")
