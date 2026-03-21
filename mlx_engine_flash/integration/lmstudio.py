
import functools
import pathlib
from typing import Any

from ..config import FlashConfig

_ORIGINAL_LOAD = None
_ORIGINAL_STREAM_GENERATE = None
_ORIGINAL_GENERATE = None
_LAST_LOOP = None

def apply_flash_patch(config: FlashConfig | None = None) -> None:
    """
    Monkey-patch mlx_lm to be Flash-compatible.
    """
    global _ORIGINAL_LOAD, _ORIGINAL_STREAM_GENERATE, _ORIGINAL_GENERATE

    if config is None:
        config = FlashConfig(enabled=False)

    try:
        import mlx_lm
        import mlx.core as mx
    except ImportError as e:
        raise ImportError("mlx_lm not installed") from e

    if _ORIGINAL_LOAD is not None:
        return

    # LOCK DOWN METAL LIMITS
    mx.metal.set_cache_limit = lambda *args, **kwargs: None
    mx.metal.set_wired_limit = lambda *args, **kwargs: None

    _ORIGINAL_LOAD = mlx_lm.load
    _ORIGINAL_STREAM_GENERATE = mlx_lm.stream_generate
    _ORIGINAL_GENERATE = getattr(mlx_lm, "generate", None)

    # 1. Patch LOAD
    def _flash_load(path, **kwargs):
        if not _should_use_flash(str(path), config):
            return _ORIGINAL_LOAD(path, **kwargs)
        
        from ..generation import FlashGenerationLoop
        from mlx_lm.utils import load_tokenizer
        from pathlib import Path
        
        global _LAST_LOOP
        
        # Return a FlashGenerationLoop proxy + tokenizer
        # The proxy's .stream_generate() runs synchronous per-layer execution
        tokenizer = load_tokenizer(Path(path))
        _LAST_LOOP = FlashGenerationLoop(Path(path), config)
        return _LAST_LOOP, tokenizer  # satisfies (model, tokenizer) contract

    mlx_lm.load = _flash_load

    # 2. Patch stream_generate
    def _flash_stream_generate(model, tokenizer, prompt, **kwargs):
        from ..generation import FlashGenerationLoop
        if isinstance(model, FlashGenerationLoop):
            # Use our synchronous loop
            yield from model.stream_generate(prompt, **kwargs)
        else:
            yield from _ORIGINAL_STREAM_GENERATE(model, tokenizer, prompt, **kwargs)

    mlx_lm.stream_generate = _flash_stream_generate

    # 3. Patch generate (optional but good for completeness)
    if _ORIGINAL_GENERATE:
        def _flash_generate(model, tokenizer, prompt, **kwargs):
            from ..generation import FlashGenerationLoop
            if isinstance(model, FlashGenerationLoop):
                # Just join the stream
                return "".join(list(model.stream_generate(prompt, **kwargs)))
            else:
                return _ORIGINAL_GENERATE(model, tokenizer, prompt, **kwargs)
        mlx_lm.generate = _flash_generate


def remove_flash_patch() -> None:
    """Restore the original state."""
    global _ORIGINAL_LOAD, _ORIGINAL_STREAM_GENERATE, _ORIGINAL_GENERATE, _LAST_LOOP
    if _ORIGINAL_LOAD is None: return
    
    import mlx_lm
    mlx_lm.load = _ORIGINAL_LOAD
    mlx_lm.stream_generate = _ORIGINAL_STREAM_GENERATE
    if _ORIGINAL_GENERATE:
        mlx_lm.generate = _ORIGINAL_GENERATE
        
    if _LAST_LOOP and hasattr(_LAST_LOOP, "manager"):
        _LAST_LOOP.manager.shutdown()
    
    _ORIGINAL_LOAD = None
    _ORIGINAL_STREAM_GENERATE = None
    _ORIGINAL_GENERATE = None
    _LAST_LOOP = None


def _should_use_flash(model_path: str, config: FlashConfig) -> bool:
    if config.enabled: return True
    p = pathlib.Path(model_path)
    for mf_name in ("Modelfile", "modelfile"):
        mf_path = p / mf_name
        if mf_path.exists():
            from .modelfile import parse_flash_directives
            mf_config = parse_flash_directives(mf_path.read_text())
            return mf_config.enabled
    return config.enabled
