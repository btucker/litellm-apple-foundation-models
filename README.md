## LiteLLM Apple Foundation Models (custom provider)

LiteLLM custom provider for Apple's on-device Foundation Models (macOS 26+ with Apple Intelligence).

### Pre-requisites

- macOS 26.0+ (Sequoia) with Apple Intelligence enabled
- Python 3.9+

### Install

```bash
pip install litellm-apple-foundation-models
```

Requires Python 3.9+ and `apple-foundation-models` (only available on macOS).

### Quick start

```python
import litellm
from litellm_apple_foundation_models import (
    register_apple_foundation_models_provider,
)

# Register the provider once in your process
register_apple_foundation_models_provider()

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

Async:

```python
import asyncio
from litellm import acompletion

async def main():
    response = await acompletion(
        model="apple_foundation_models/system",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        max_tokens=100,
    )
    print(response)

asyncio.run(main())
```

Async + streaming:

```python
import asyncio
from litellm import acompletion

async def main():
    response = await acompletion(
        model="apple_foundation_models/system",
        messages=[{"role": "user", "content": "Write a short poem about AI"}],
        stream=True,
        max_tokens=200,
    )

    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

asyncio.run(main())
```

### Tool calling

```python
from litellm import completion
from litellm_apple_foundation_models import register_apple_foundation_models_provider

register_apple_foundation_models_provider()

def get_weather(location: str, units: str = "celsius") -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: 22°{units[0].upper()}, sunny"

def calculate(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "What's the weather in Paris and what's 5 plus 7?"}],
    tool_functions=[get_weather, calculate],
    max_tokens=200,
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Tool: {tool_call.function.name}({tool_call.function.arguments})")

print(response.choices[0].message.content)
```

You can also pass OpenAI-style schemas via `tools=[...]` and map implementations with `tool_functions`.

### Structured output

Pydantic:

```python
from pydantic import BaseModel
from litellm import completion

class Person(BaseModel):
    name: str
    age: int
    city: str

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "Extract person info: Alice is 30 and lives in Paris."}],
    response_format=Person,
    max_tokens=150,
)
print(response.choices[0].message.content)
```

JSON schema:

```python
import json
from litellm import completion

schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["answer"],
}

response = completion(
    model="apple_foundation_models/system",
    messages=[{"role": "user", "content": "Is the sky blue? Return JSON with answer and confidence (0-1)."}],
    response_format={"type": "json_schema", "json_schema": {"schema": schema}},
    max_tokens=100,
)
print(response.choices[0].message.content)
```

### Supported parameters

- `temperature`: float
- `max_tokens`: int
- `stream`: bool
- `tools` / `tool_functions`
- `response_format` (Pydantic model or JSON schema)

### How it works

- Uses LiteLLM's `CustomLLM` interface to avoid touching LiteLLM core.
- Registers itself via `litellm.custom_provider_map` so calls to `litellm.completion` with `model="apple_foundation_models/*"` are routed here.

### Development

- Tests live under `tests/` (mirrors the coverage from the original core PR).
- To run tests locally:

```bash
pip install -e .[dev]
pytest
```
