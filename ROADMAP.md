# mlx-flash Roadmap ⚡

This document outlines the planned milestones for `mlx-flash` as it moves from beta to a stable, production-grade utility.

## v0.1.x: Stability & Polished Beta
*Focus: Fixing the "paper cuts" and ensuring 100% correctness.*

- [ ] **PyPI Release**: Official package distribution for easier installation.
- [ ] **CI Benchmarking**: Automatic performance regression tracking in GitHub Actions.
- [ ] **Sampler Parity**: Ensure 100% numerical parity with standard `mlx-lm` for all sampling parameters.
- [ ] **Improved Diagnostics**: More granular RAM profiling in `flash-monitor`.

## v0.2.0: Background I/O & Disk KV
*Focus: Eliminating I/O latency and handling massive contexts.*

- [ ] **Background I/O Thread**: Implement real async prefetch by parsing Safetensors headers and issuing `madvise(WILLNEED)` on a separate thread using raw file mmap.
- [ ] **Stable Disk KV Cache**: Robust offloading and reloading of KV cache entries from SSD with full data integrity checks.
- [ ] **Adaptive Budgeting**: Automatically adjust RAM limits based on real-time system pressure signals from the OS.

## v0.3.0: High-Performance MoE Weight Streaming
*Focus: Making massive MoE models (Mixtral, DeepSeek) run at 10+ tok/s.*

- [ ] **Expert Prefetching**: Predictively load the next top-K experts while the current experts are executing.
- [ ] **Layer Skipping Support**: Improved logic for architectures that support dynamic layer execution.
- [ ] **Multi-GPU / Multi-Node (Experimental)**: Exploring streaming across multiple unified memory pools.

---

## v1.0.0: Native Integration
*Focus: Moving from a monkey-patch to a standard feature.*

- [ ] **Upstream PR to `mlx-lm`**: Propose native `FlashLLM` support to eliminate the need for monkey-patching.
- [ ] **Official LM Studio Integration**: Seamless "Flash" checkbox in the LM Studio UI.
- [ ] **Documentation**: Comprehensive API reference and integration guides for other frameworks.
