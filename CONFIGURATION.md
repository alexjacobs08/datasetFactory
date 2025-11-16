# Configuration Guide

DatasetFactory uses a flexible configuration system that supports both YAML config files and environment variables.

## Quick Start

### 1. Use Preset Configurations

Copy a preset config to your project:

```bash
# For Groq (high speed, low cost)
cp generation_config_groq.yaml generation_config.yaml

# For Gemini (balanced, free preview)
cp generation_config_gemini.yaml generation_config.yaml

# For OpenAI (high quality)
cp generation_config_openai.yaml generation_config.yaml
```

Then just run your code - config is automatically loaded!

### 2. Set API Keys

Create a `.env` file with your API keys:

```bash
# Only the keys for providers you're using
GEMINI_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

## Configuration Reference

### Complete YAML Example

```yaml
# generation_config.yaml

models:
  # Primary model for most tasks (fast, cheap)
  default_model: gemini:gemini-2.0-flash-exp
  
  # Medium quality model for balanced tasks
  medium_model: openai:gpt-4o-mini
  
  # Premium model for complex tasks
  premium_model: gemini:gemini-1.5-flash
  
  # Temperature (0.0 = deterministic, 2.0 = very creative)
  temperature: 0.7

rate_limiting:
  # Enable automatic rate limiting
  enabled: true
  
  # Requests per minute (null = auto-detect)
  rpm: null
  
  # Tokens per minute (null = auto-detect)
  tpm: null
  
  # Max retries on rate limit (null = auto-detect: 8 for Groq, 3 for others)
  max_retries: null
  
  # Initial backoff time in seconds
  initial_backoff: 1.0
  
  # Maximum backoff time in seconds
  max_backoff: 60.0

concurrency:
  # Maximum parallel document generations
  max_concurrent_generations: 10
  
  # Batch size for processing
  batch_size: 10
```

## Configuration Loading Priority

The system loads configuration in this order (first found wins):

1. **Explicitly provided config instance**
   ```python
   from dataset_factory.config import GenerationConfig
   from dataset_factory import DatasetGenerator
   from dataset_factory.llm import LLMClient
   
   config = GenerationConfig.load("my_config.yaml")
   client = LLMClient(config=config)
   generator = DatasetGenerator(llm_client=client)
   ```

2. **Config file at specified path**
   ```python
   client = LLMClient(config_path="configs/groq.yaml")
   ```

3. **generation_config.yaml in current directory**
   ```bash
   ./generation_config.yaml
   ```

4. **generation_config.yaml in workspace root**
   - Searches up directory tree for common indicators (pyproject.toml, .git, README.md)

5. **Environment variables**
   ```bash
   DEFAULT_MODEL=groq:llama-3.1-8b-instant
   TEMPERATURE=0.8
   MAX_CONCURRENT_GENERATIONS=15
   ```

6. **Built-in defaults**
   - Gemini 2.0 Flash as default model
   - Auto-configured rate limiting
   - 10 concurrent generations

## Model Configuration

### Available Models

#### Groq (High Speed, Low Cost)
```yaml
models:
  default_model: groq:llama-3.1-8b-instant  # 840 TPS, $0.05/$0.08 per M tokens
  default_model: groq:llama-3.3-70b-versatile  # Better quality
```

#### Google Gemini (Balanced)
```yaml
models:
  default_model: gemini:gemini-2.0-flash-exp  # Free during preview
  premium_model: gemini:gemini-1.5-pro  # Higher quality
```

#### OpenAI (High Quality)
```yaml
models:
  default_model: openai:gpt-4o-mini  # Balanced
  premium_model: openai:gpt-4o  # Best quality
```

#### Anthropic
```yaml
models:
  premium_model: anthropic:claude-3-5-sonnet-20241022
```

### Temperature Settings

```yaml
models:
  temperature: 0.0   # Deterministic, consistent outputs
  temperature: 0.7   # Balanced (recommended)
  temperature: 1.5   # Creative, varied outputs
```

## Rate Limiting

### Automatic Configuration

When `rpm` and `tpm` are `null`, the system auto-detects based on provider:

| Provider | Model | Auto RPM | Auto TPM |
|----------|-------|----------|----------|
| Groq | llama-3.1-8b-instant | 30 | 200,000 |
| Groq | llama-3.3-70b | 30 | 24,000 |
| Gemini | All models | 60 | 500,000 |
| OpenAI | All models | 60 | 500,000 |

### Manual Configuration

Override auto-detection for specific needs:

```yaml
rate_limiting:
  enabled: true
  rpm: 20        # Lower for conservative approach
  tpm: 150000    # 75% of Groq's 200k limit
  max_retries: 10  # More retries if needed
```

### Disable Rate Limiting

```yaml
rate_limiting:
  enabled: false
```

**Warning:** Only disable for providers with very high limits or local models.

## Concurrency Settings

### Max Concurrent Generations

Controls parallel document generation:

```yaml
concurrency:
  max_concurrent_generations: 5   # Conservative (OpenAI)
  max_concurrent_generations: 10  # Balanced (Gemini)
  max_concurrent_generations: 15  # Aggressive (Groq)
  max_concurrent_generations: 20  # Maximum (local models)
```

**Recommendations:**
- **Groq**: 10-15 (high throughput, 840 TPS)
- **Gemini**: 10 (good default)
- **OpenAI**: 5-10 (depends on tier/cost)
- **Local models**: 20+ (no rate limits)

### Batch Size

Currently used for internal batching:

```yaml
concurrency:
  batch_size: 10  # Default, rarely needs changing
```

## Provider-Specific Configs

### Groq (Recommended for Speed)

```yaml
models:
  default_model: groq:llama-3.1-8b-instant
  temperature: 0.7

rate_limiting:
  enabled: true
  rpm: 30
  tpm: 200000
  max_retries: 8

concurrency:
  max_concurrent_generations: 15
```

**Cost:** ~$0.46 per 1K documents

### Gemini (Recommended for Balance)

```yaml
models:
  default_model: gemini:gemini-2.0-flash-exp
  temperature: 0.7

rate_limiting:
  enabled: true
  rpm: null  # Auto-detect
  tpm: null  # Auto-detect
  max_retries: 3

concurrency:
  max_concurrent_generations: 10
```

**Cost:** Free during preview (normally ~$0.80 per 1K documents)

### OpenAI (Recommended for Quality)

```yaml
models:
  default_model: openai:gpt-4o-mini
  temperature: 0.7

rate_limiting:
  enabled: true
  rpm: 30
  tpm: 200000
  max_retries: 3

concurrency:
  max_concurrent_generations: 5
```

**Cost:** ~$2-5 per 1K documents (depending on tier)

## Programmatic Configuration

### Load and Modify Config

```python
from dataset_factory.config import GenerationConfig

# Load from file
config = GenerationConfig.load("generation_config.yaml")

# Modify
config.models.temperature = 0.8
config.concurrency.max_concurrent_generations = 20

# Save
config.save("custom_config.yaml")
```

### Create from Scratch

```python
from dataset_factory.config import GenerationConfig

config = GenerationConfig(
    models={
        "default_model": "groq:llama-3.1-8b-instant",
        "temperature": 0.7,
    },
    rate_limiting={
        "enabled": True,
        "rpm": 30,
        "tpm": 200000,
    },
    concurrency={
        "max_concurrent_generations": 15,
    }
)

# Use directly
from dataset_factory.llm import LLMClient
from dataset_factory import DatasetGenerator

client = LLMClient(config=config)
generator = DatasetGenerator(llm_client=client)
```

### Environment Variables (Legacy)

Still supported for backward compatibility:

```python
import os

os.environ["DEFAULT_MODEL"] = "groq:llama-3.1-8b-instant"
os.environ["TEMPERATURE"] = "0.8"
os.environ["MAX_CONCURRENT_GENERATIONS"] = "15"
os.environ["RATE_LIMIT_RPM"] = "30"
os.environ["RATE_LIMIT_TPM"] = "200000"

# Config automatically loaded from env vars
generator = DatasetGenerator()
```

## Troubleshooting

### Config Not Loading

1. Check file exists: `ls generation_config.yaml`
2. Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('generation_config.yaml'))"`
3. Enable debug: Look for "✓ Groq rate limiting" or similar messages

### Rate Limits Still Hit

1. Lower `max_concurrent_generations`
2. Set more conservative `tpm` (e.g., 70% of limit)
3. Increase `max_retries`
4. Check your API tier/plan limits

### Wrong Model Used

Check config priority order. Explicit config always wins:

```python
# This WILL use the config file:
generator = DatasetGenerator()

# This WILL override with env var:
os.environ["DEFAULT_MODEL"] = "openai:gpt-4o"
generator = DatasetGenerator()
```

## Best Practices

1. **Use config files** instead of env vars for generation settings
2. **Keep API keys in .env** for security (never commit!)
3. **Start with presets** and customize as needed
4. **Use null for auto-detection** unless you have specific needs
5. **Test with small batches** before large generations
6. **Monitor costs** with the built-in cost tracking

## Examples

See the `examples/` directory:
- `config_file_example.py` - Complete config usage examples
- `rate_limiting_example.py` - Rate limiting demonstrations
- `groq_example.py` - Groq-specific configuration

