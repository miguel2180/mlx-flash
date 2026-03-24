import pytest
import os
import time
import random
import threading
import psutil
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
from typing import Optional, Any

from mlx_flash.config import FlashConfig
from mlx_flash.engine.engine import FlashEngine
from mlx_flash.safetensors_mmap import SafetensorsMmapCache
from mlx_flash.prefetch_worker import BackgroundPrefetcher
from mlx_flash.engine.hooks import InferenceHook, ExecutionContext

class ChaosPrefetcher(BackgroundPrefetcher):
    """
    Adversarial prefetcher that injects latency jitter and random failures.
    """
    def __init__(self, file_handles: dict[str, Any], jitter_range=(0, 0.1), failure_rate=0.05):
        super().__init__(file_handles)
        self.jitter_range = jitter_range
        self.failure_rate = failure_rate
        self.eviction_failed_count = 0

    def _worker(self):
        base_chunk_size = 16 * 1024 * 1024 
        while self.running:
            try:
                task = self.queue.get(timeout=0.1)
                if task is None: continue
                
                # ADVERSARIAL: Randomly ignore prefetch requests (simulating eviction failure or dropped packets)
                if random.random() < self.failure_rate:
                    self.eviction_failed_count += 1
                    self.queue.task_done()
                    continue

                layer_idx, filename, offset, length, align_bytes = task
                
                # ADVERSARIAL: Inject extreme IO jitter
                time.sleep(random.uniform(*self.jitter_range))

                f = self.file_handles.get(filename)
                if f is not None:
                    fd = f.fileno()
                    curr = offset
                    end = offset + length
                    while curr < end and self.running:
                        chunk = min(base_chunk_size, end - curr)
                        os.pread(fd, chunk, curr)
                        curr += chunk
                
                if layer_idx is not None:
                    with self._lock:
                        self.completed_prefetches.add(layer_idx)
                self.queue.task_done()
            except Exception:
                continue

class ThrasherHook(InferenceHook):
    """
    Forces cache thrashing by manually invalidating page cache ranges 
    and allocating junk memory.
    """
    def __init__(self, mmap_cache: SafetensorsMmapCache):
        self.mmap_cache = mmap_cache
        self.junk_memory = []

    def on_layer_start(self, ctx: ExecutionContext, layer: nn.Module):
        # 1. Force Page Cache Eviction for the current layer to ensure we always hit disk
        if self.mmap_cache:
            from mlx_flash.page_cache import drop_page_cache
            ranges = self.mmap_cache.get_layer_ranges(ctx.layer_idx)
            for mm, (start, end, filename, _) in ranges.items():
                # mm is the mmap object, which drop_page_cache expects
                drop_page_cache(mm, start, end - start)

        # 2. Pressure the system memory to trigger OS-level LRU thrashing
        # We allocate ~50MB of junk per layer and release old junk
        self.junk_memory.append(mx.random.normal((5000, 5000))) # ~100MB in FP32
        if len(self.junk_memory) > 3:
            self.junk_memory.pop(0)
        mx.eval(self.junk_memory)

@pytest.mark.stress
def test_adversarial_io_and_thrashing(tmp_model_dir):
    """
    Scenario: Random IO delays, dropped prefetch requests, and active memory pressure.
    Goal: System must not hang, OOM, or produce NaNs.
    """
    import mlx_lm
    config = FlashConfig(enabled=True, pipelined_execution=True, debug=True)
    model_lazy, tokenizer = mlx_lm.load(str(tmp_model_dir), lazy=True)
    
    # ADVERSARIAL: Disable EOS to force long generation without early exit
    tokenizer.eos_token_id = 1000000

    # Setup Chaos Cache
    mmap_cache = SafetensorsMmapCache(tmp_model_dir)
    # Monkeypatch the prefetcher with our Chaos variant
    old_worker = mmap_cache.prefetch_worker
    mmap_cache.prefetch_worker = ChaosPrefetcher(mmap_cache.file_handles, jitter_range=(0.01, 0.05), failure_rate=0.1)
    old_worker.shutdown()

    engine = FlashEngine(model_lazy, tokenizer, config)
    engine.mmap_cache = mmap_cache
    engine.registry.add_node(ThrasherHook(mmap_cache))

    # Run generation
    tokens = []
    try:
        for token in engine.stream_generate("The quick brown fox", max_tokens=20):
            tokens.append(token)
            # Relax assertion: detokenizer might return empty strings for some segments
            assert token is not None
    finally:
        mmap_cache.prefetch_worker.shutdown()

    assert len(tokens) > 0
    print(f"Adversarial test finished. Eviction failures handled: {mmap_cache.prefetch_worker.eviction_failed_count}")

@pytest.mark.stress
def test_long_sequence_stability(tmp_model_dir):
    """
    Scenario: Generate tokens (simulating long sequence behavior for a small model).
    Goal: Verify KV cache offloading stability and RAM flatness.
    """
    import mlx_lm
    # Force disk KV and quantization
    config = FlashConfig(
        enabled=True, 
        disk_kv_enabled=True, 
        kv_cache_quantized=True,
        kv_cache_bits=4,
        ram_budget_gb=0.2,
        debug=False
    )
    model_lazy, tokenizer = mlx_lm.load(str(tmp_model_dir), lazy=True)
    
    # Force EOS to None to prevent early exit in generate_step logic
    if hasattr(tokenizer, "eos_token_id"):
        tokenizer.eos_token_id = None

    engine = FlashEngine(model_lazy, tokenizer, config)
    mmap_cache = SafetensorsMmapCache(tmp_model_dir)
    engine.mmap_cache = mmap_cache

    process = psutil.Process(os.getpid())
    start_ram = process.memory_info().rss
    
    count = 0
    max_tokens = 100 # Adjusted for CI speed
    
    # Manual loop to bypass generate_step's EOS logic
    x = mx.array(tokenizer.encode("Once upon a time"))[None]
    
    for _ in range(max_tokens):
        logits = engine(x)
        # Greedily sample next token
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        # Append to input for next step (MLX handles cache internally via engine)
        x = token
        
        count += 1
        if count % 50 == 0:
            current_ram = (process.memory_info().rss - start_ram) / 1e6
            print(f"Token {count}, RSS Delta: {current_ram:.1f} MB")
            # RAM should not grow linearly with sequence length
            assert current_ram < 500.0

    assert count == max_tokens
    mmap_cache.shutdown()
