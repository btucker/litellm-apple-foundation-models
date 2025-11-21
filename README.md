## LiteLLM Apple Foundation Models (custom provider)

Standalone LiteLLM custom provider for Apple's on-device Foundation Models (macOS 26+ with Apple Intelligence).

### Install

```bash
pip install -e .
```

Requires Python 3.9+ and `apple-foundation-models` (only available on macOS).

### Quick start

```python
import litellm
from litellm_apple_foundation_models import register_provider

# Register the provider once in your process
register_provider()

resp = litellm.completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "Hello from macOS"}],
)
print(resp.choices[0].message.content)
```

Streaming and async are supported:

```python
resp = litellm.completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "Write a haiku about objc"}],
    stream=True,
)
for chunk in resp:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### How it works

- Reuses LiteLLM's `CustomLLM` interface to avoid touching LiteLLM core.
- Wraps the Apple SDK (`apple-foundation-models`) with the same transformation logic that was being added to core.
- Registers itself via `litellm.custom_provider_map` so calls to `litellm.completion` with `model="apple_foundation_models/*"` are routed here.

### Development

- Tests live under `tests/` (mirrors the coverage from the original core PR).
- To run tests locally:

```bash
pip install -e .[dev]
pytest
```
