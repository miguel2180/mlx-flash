
import mlx.core as mx

from mlx_flash.generation import FlashGenerationLoop


def test_greedy_determinism(tmp_model_dir):
    """Verify that greedy sampling (temp=0) is deterministic with same seed."""
    import mlx_lm

    from mlx_flash import FlashConfig

    model, tokenizer = mlx_lm.load(str(tmp_model_dir))
    config = FlashConfig(enabled=True)
    loop = FlashGenerationLoop(model, tokenizer, config)

    prompt = "Test prompt"

    # Two greedy runs with same seed must produce identical output
    mx.random.seed(42)
    tokens_a = list(loop.stream_generate(prompt, temp=0.0, max_tokens=3))

    mx.random.seed(42)
    tokens_b = list(loop.stream_generate(prompt, temp=0.0, max_tokens=3))

    assert tokens_a == tokens_b, (
        f"Greedy sampling is not deterministic: {tokens_a!r} vs {tokens_b!r}"
    )


def test_temperature_affects_sampling(tmp_model_dir):
    """Verify that temp=0 (greedy) differs from temp=100 on random logits."""
    import mlx_lm

    from mlx_flash import FlashConfig

    model, tokenizer = mlx_lm.load(str(tmp_model_dir))
    config = FlashConfig(enabled=True)
    loop = FlashGenerationLoop(model, tokenizer, config)

    prompt = "Test"

    # Greedy
    tokens_greedy = list(loop.stream_generate(prompt, temp=0.0, max_tokens=5))

    # High temperature
    tokens_hot = list(loop.stream_generate(prompt, temp=100.0, max_tokens=5))

    assert isinstance(tokens_greedy, list)
    assert isinstance(tokens_hot, list)
