"""LLM client wrapper using pydantic-ai for model abstraction and cost tracking."""

import asyncio
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from genai_prices import Usage, calc_price
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from dataset_factory.config.generation_config import GenerationConfig

# Import for type hint, but make it optional
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dataset_factory.cost_tracker import CostTracker

# Load environment variables
load_dotenv()

T = TypeVar("T")


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API requests.
    
    Supports both requests-per-minute (RPM) and tokens-per-minute (TPM) limits.
    """
    
    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> None:
        """
        Initialize rate limiter.
        
        Args:
            rpm: Requests per minute limit
            tpm: Tokens per minute limit
        """
        self.rpm = rpm
        self.tpm = tpm
        
        # Request tracking
        self.request_tokens = rpm if rpm else float('inf')
        self.request_capacity = rpm if rpm else float('inf')
        self.request_last_update = time.time()
        
        # Token tracking
        self.token_tokens = tpm if tpm else float('inf')
        self.token_capacity = tpm if tpm else float('inf')
        self.token_last_update = time.time()
        
        self.lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """
        Acquire tokens before making a request.
        
        Args:
            estimated_tokens: Estimated tokens for this request
        """
        async with self.lock:
            await self._wait_for_capacity(1, estimated_tokens)
    
    async def _wait_for_capacity(self, requests: int, tokens: int) -> None:
        """Wait until we have capacity for this request."""
        while True:
            current_time = time.time()
            
            # Refill request bucket
            if self.rpm:
                time_passed = current_time - self.request_last_update
                refill = (time_passed / 60.0) * self.request_capacity
                self.request_tokens = min(
                    self.request_capacity,
                    self.request_tokens + refill
                )
                self.request_last_update = current_time
            
            # Refill token bucket
            if self.tpm:
                time_passed = current_time - self.token_last_update
                refill = (time_passed / 60.0) * self.token_capacity
                self.token_tokens = min(
                    self.token_capacity,
                    self.token_tokens + refill
                )
                self.token_last_update = current_time
            
            # Check if we have capacity
            has_request_capacity = self.request_tokens >= requests
            has_token_capacity = self.token_tokens >= tokens
            
            if has_request_capacity and has_token_capacity:
                # Consume tokens
                self.request_tokens -= requests
                self.token_tokens -= tokens
                return
            
            # Calculate wait time
            wait_time = 0.0
            
            if not has_request_capacity and self.rpm:
                tokens_needed = requests - self.request_tokens
                wait_time = max(wait_time, (tokens_needed / self.request_capacity) * 60.0)
            
            if not has_token_capacity and self.tpm:
                tokens_needed = tokens - self.token_tokens
                wait_time = max(wait_time, (tokens_needed / self.token_capacity) * 60.0)
            
            # Wait a bit before checking again
            await asyncio.sleep(min(wait_time + 0.1, 1.0))


@dataclass
class UsageStats:
    """Usage statistics for a phase."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0

    def add(self, other: "UsageStats") -> None:
        """Add another UsageStats to this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests
        self.cost_usd += other.cost_usd


@dataclass
class CostBreakdown:
    """Cost breakdown by phase."""

    config_generation: UsageStats
    world_building: UsageStats
    document_generation: UsageStats
    query_generation: UsageStats
    total_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config_generation": {
                "tokens": self.config_generation.input_tokens
                + self.config_generation.output_tokens,
                "cost_usd": round(self.config_generation.cost_usd, 2),
            },
            "world_building": {
                "tokens": self.world_building.input_tokens + self.world_building.output_tokens,
                "cost_usd": round(self.world_building.cost_usd, 2),
            },
            "document_generation": {
                "tokens": self.document_generation.input_tokens
                + self.document_generation.output_tokens,
                "cost_usd": round(self.document_generation.cost_usd, 2),
            },
            "query_generation": {
                "tokens": self.query_generation.input_tokens
                + self.query_generation.output_tokens,
                "cost_usd": round(self.query_generation.cost_usd, 2),
            },
            "total_cost_usd": round(self.total_cost_usd, 2),
        }


class LLMClient:
    """
    LLM client wrapper for pydantic-ai with cost tracking.

    Supports model selection (premium vs default) and automatic cost calculation
    using genai-prices for up-to-date pricing across all supported models.
    """

    def __init__(
        self,
        config: GenerationConfig | None = None,
        config_path: str | Path | None = None,
        cost_tracker: "CostTracker | None" = None,
        validate_keys: bool = True,
    ) -> None:
        """
        Initialize LLM client with configuration.
        
        Args:
            config: Optional GenerationConfig instance
            config_path: Optional path to config file
            cost_tracker: Optional CostTracker for batch-level tracking to files
            validate_keys: Whether to validate API keys on startup (default: True)
            
        Priority:
            1. Provided config instance
            2. Config file at config_path
            3. generation_config.yaml in current/workspace directory
            4. Environment variables
            5. Defaults
        """
        # Load configuration
        if config is None:
            config = GenerationConfig.load(config_path)
        
        self.config = config
        self.external_cost_tracker = cost_tracker
        
        # Validate API keys if requested
        if validate_keys:
            self._validate_api_keys()
        
        # Model names should be in format 'provider:model-name' for pydantic-ai
        self.default_model = config.default_model.model
        self.medium_model = config.medium_model.model
        self.premium_model = config.premium_model.model
        
        # Store model tier configs for per-model settings
        self._model_tiers = {
            "default": config.default_model,
            "medium": config.medium_model,
            "premium": config.premium_model,
        }
        
        # Use default model's settings as fallback for backward compatibility
        self.temperature = config.default_model.temperature
        
        # Determine max_retries for default model (for backward compatibility)
        if config.default_model.rate_limiting.max_retries is not None:
            self.max_retries = config.default_model.rate_limiting.max_retries
        elif "groq:" in self.default_model.lower():
            self.max_retries = 8
        else:
            self.max_retries = 3

        # Rate limiting setup - per model tier
        self.rate_limiters: dict[str, TokenBucketRateLimiter | None] = {}
        self._setup_rate_limiting()

        # Cost tracking by phase
        self.cost_tracker = {
            "config_generation": UsageStats(),
            "world_building": UsageStats(),
            "document_generation": UsageStats(),
            "query_generation": UsageStats(),
        }

    def _validate_api_keys(self) -> None:
        """
        Validate that required API keys are present for configured models.
        
        Raises:
            ValueError: If required API key is missing
        """
        # Check all three model tiers
        models_to_check = [
            ("default", self.config.default_model.model),
            ("medium", self.config.medium_model.model),
            ("premium", self.config.premium_model.model),
        ]
        
        missing_keys = []
        
        for tier, model in models_to_check:
            provider = model.split(":")[0].lower() if ":" in model else "unknown"
            
            # Map provider to expected env var
            key_map = {
                "groq": "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "google": "GEMINI_API_KEY",
                "google-gla": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            
            expected_key = key_map.get(provider)
            if expected_key and not os.getenv(expected_key):
                missing_keys.append((tier, model, expected_key))
        
        if missing_keys:
            print(f"\n{'='*70}")
            print(f"❌ Missing API Key(s)")
            print(f"{'='*70}")
            for tier, model, key_name in missing_keys:
                print(f"  {tier} model ({model}) requires: {key_name}")
            print(f"\n💡 Solutions:")
            print(f"  1. Create a .env file with your API key(s)")
            print(f"  2. Or export them: export {missing_keys[0][2]}=your_key_here")
            print(f"  3. Or change your model in generation_config.yaml")
            print(f"\n  Get API keys from:")
            print(f"    - Groq: https://console.groq.com")
            print(f"    - Gemini: https://aistudio.google.com")
            print(f"    - OpenAI: https://platform.openai.com")
            print(f"    - Anthropic: https://console.anthropic.com")
            print(f"{'='*70}\n")
            raise ValueError(f"Missing required API key: {missing_keys[0][2]}")

    def _get_auto_rate_limits(self, model: str) -> tuple[int, int]:
        """
        Auto-detect rate limits based on model provider.
        
        Returns:
            (rpm, tpm) tuple
        """
        model_lower = model.lower()
        
        if "groq:" in model_lower:
            # Groq has strict TPM limits
            if "llama-3.1-8b-instant" in model_lower:
                return (30, int(250_000 * 0.8))  # 200k TPM
            elif "llama-3.3-70b" in model_lower:
                return (30, int(30_000 * 0.8))
            else:
                return (30, int(100_000 * 0.8))
        else:
            # Other providers typically have higher limits
            return (60, 500_000)

    def _setup_rate_limiting(self) -> None:
        """Set up rate limiting for each model tier based on config."""
        # Check if rate limiting is globally disabled
        if not self.config.rate_limiting.enabled:
            for tier in ["default", "medium", "premium"]:
                self.rate_limiters[tier] = None
            print("⚠️  Rate limiting disabled globally")
            return
        
        # Set up rate limiting for each model tier
        for tier_name, tier_config in self._model_tiers.items():
            rpm = tier_config.rate_limiting.rpm
            tpm = tier_config.rate_limiting.tpm
            
            # Auto-detect if not specified
            if rpm is None or tpm is None:
                auto_rpm, auto_tpm = self._get_auto_rate_limits(tier_config.model)
                rpm = rpm or auto_rpm
                tpm = tpm or auto_tpm
            
            self.rate_limiters[tier_name] = TokenBucketRateLimiter(rpm=rpm, tpm=tpm)
            
            # Print info for default model
            if tier_name == "default":
                provider = tier_config.model.split(":")[0] if ":" in tier_config.model else "unknown"
                print(f"✓ Rate limiting ({provider}): {rpm} RPM, {tpm:,} TPM")
    
    def _get_rate_limiter(self, model_type: str) -> TokenBucketRateLimiter | None:
        """Get rate limiter for specific model type."""
        return self.rate_limiters.get(model_type)
    
    def _get_max_retries(self, model_type: str) -> int:
        """Get max retries for specific model type."""
        tier_config = self._model_tiers.get(model_type)
        if tier_config and tier_config.rate_limiting.max_retries is not None:
            return tier_config.rate_limiting.max_retries
        
        # Auto-detect based on model
        model = self._select_model(model_type)
        if "groq:" in model.lower():
            return 8
        return 3
    
    def _get_temperature(self, model_type: str) -> float:
        """Get temperature for specific model type."""
        tier_config = self._model_tiers.get(model_type)
        if tier_config:
            return tier_config.temperature
        return self.temperature

    def _select_model(self, model_type: str) -> str:
        """
        Select model based on type.
        
        Args:
            model_type: 'default', 'medium', or 'premium'
            
        Returns:
            Model string for pydantic-ai
        """
        if model_type == "premium":
            return self.premium_model
        elif model_type == "medium":
            return self.medium_model
        else:
            return self.default_model

    def _calculate_cost(self, usage: Any, model: str) -> float:
        """
        Calculate cost in USD from usage stats using genai-prices.
        
        Falls back to manual pricing for Groq and other models not in genai-prices.
        
        Args:
            usage: Usage object with input_tokens and output_tokens
            model: Model name in format 'provider:model-name' (pydantic-ai format)
        """
        # Manual pricing for Groq models (not in genai-prices yet)
        groq_pricing = {
            "llama-3.1-8b-instant": (0.05, 0.08),
            "llama-3.3-70b-versatile": (0.59, 0.79),
            "llama-4-scout": (0.11, 0.34),
            "llama-4-maverick": (0.20, 0.60),
            "llama-guard-4-12b": (0.20, 0.20),
            "qwen3-32b": (0.29, 0.59),
            "gpt-oss-20b": (0.075, 0.30),
            "gpt-oss-120b": (0.15, 0.60),
            "kimi-k2-0905-1t": (1.00, 3.00),
        }
        
        # Extract model name without provider
        model_name = model.split(":", 1)[-1] if ":" in model else model
        
        # Check if it's a Groq model
        if model_name in groq_pricing:
            input_price_per_million, output_price_per_million = groq_pricing[model_name]
            input_cost = (usage.input_tokens / 1_000_000) * input_price_per_million
            output_cost = (usage.output_tokens / 1_000_000) * output_price_per_million
            return input_cost + output_cost
        
        # Try genai-prices for other models
        try:
            genai_usage = Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            
            # Try with full model string first
            try:
                price_calc = calc_price(usage=genai_usage, model_ref=model)
                return float(price_calc.total_price)
            except Exception:
                # Try without provider prefix
                price_calc = calc_price(usage=genai_usage, model_ref=model_name)
                return float(price_calc.total_price)
                
        except Exception as e:
            # If model not found or pricing unavailable, warn loudly
            print(f"\n{'='*70}")
            print(f"⚠️  WARNING: Cost calculation failed for model '{model}'")
            print(f"{'='*70}")
            print(f"Error: {e}")
            print(f"\nThis means cost tracking will be INCOMPLETE.")
            print(f"💡 Solutions:")
            print(f"  - Check your provider dashboard for actual costs")
            print(f"  - Report missing model at: https://github.com/alexjacobs08/dataset-factory/issues")
            print(f"  - Use a model with known pricing (groq:llama-3.1-8b-instant, etc.)")
            print(f"{'='*70}\n")
            return 0.0

    def _track_usage(self, usage: Any, model: str, phase: str, duration_seconds: float | None = None) -> None:
        """Track usage for a specific phase."""
        cost = self._calculate_cost(usage, model)
        stats = UsageStats(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            requests=1,
            cost_usd=cost,
        )
        self.cost_tracker[phase].add(stats)

        # Also track in external cost tracker if provided
        if self.external_cost_tracker:
            self.external_cost_tracker.accumulate(
                phase=phase,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                duration_seconds=duration_seconds,
                cost_usd=cost,  # Pass the actual calculated cost!
                count=1
            )
        # Debug: Show if external tracker is connected
        elif hasattr(self, '_debug_tracker_once'):
            pass  # Already warned
        else:
            self._debug_tracker_once = True

        # Print running cost
        total = sum(s.cost_usd for s in self.cost_tracker.values())
        print(f"[{phase}] +${cost:.4f} (Total: ${total:.4f})")

    async def _retry_with_exponential_backoff(
        self,
        func,
        max_retries: int | None = None,
        initial_wait: float | None = None,
        max_wait: float | None = None,
    ):
        """
        Retry a function with exponential backoff on rate limit errors.
        
        Args:
            func: Async function to retry
            max_retries: Maximum number of retries (uses self.max_retries if None)
            initial_wait: Initial wait time in seconds (uses config if None)
            max_wait: Maximum wait time in seconds (uses config if None)
            
        Returns:
            Result from func
            
        Raises:
            Last exception if all retries fail
        """
        if max_retries is None:
            max_retries = self.max_retries
        if initial_wait is None:
            initial_wait = self.config.rate_limiting.initial_backoff
        if max_wait is None:
            max_wait = self.config.rate_limiting.max_backoff
        
        last_exception = None
        wait_time = initial_wait
        
        for attempt in range(max_retries + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                is_rate_limit = (
                    "rate limit" in error_str
                    or "429" in str(e)
                    or "rate_limit_exceeded" in error_str
                )
                
                if not is_rate_limit or attempt >= max_retries:
                    # Not a rate limit error or out of retries
                    raise
                
                # Extract wait time from error message if available
                import re
                wait_match = re.search(r'try again in ([\d.]+)(ms|s)', str(e))
                if wait_match:
                    extracted_wait = float(wait_match.group(1))
                    if wait_match.group(2) == 'ms':
                        extracted_wait /= 1000.0
                    wait_time = min(extracted_wait + 1.0, max_wait)
                else:
                    # Use exponential backoff with jitter
                    wait_time = min(
                        initial_wait * (2 ** attempt) + random.uniform(0, 1),
                        max_wait
                    )
                
                print(f"⚠️  Rate limit hit (attempt {attempt + 1}/{max_retries + 1}) - waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        
        # Should not reach here, but just in case
        raise last_exception

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[BaseModel],
        model_type: str = "default",
        phase: str = "general",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModel:
        """
        Generate structured output using pydantic-ai.

        Args:
            prompt: User prompt
            output_type: Pydantic model class for output
            model_type: 'default', 'medium', or 'premium'
            phase: Phase name for cost tracking
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Instance of output_type
        """
        model = self._select_model(model_type)

        # Apply model-specific rate limiting
        rate_limiter = self._get_rate_limiter(model_type)
        if rate_limiter:
            # Estimate tokens (prompt length + max_tokens or typical response)
            estimated_tokens = len(prompt.split()) * 1.3
            if max_tokens:
                estimated_tokens += max_tokens
            else:
                estimated_tokens += 500
            await rate_limiter.acquire(int(estimated_tokens))

        # For structured output, allow validation retries (3 attempts)
        # We still handle rate limit retries separately
        agent = Agent(
            model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=3,  # Allow validation retries for structured output
        )
        
        # Create model settings with model-specific temperature and max_tokens
        temperature = self._get_temperature(model_type)
        model_settings = ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async def _run_agent():
            try:
                return await agent.run(prompt, model_settings=model_settings)
            except Exception as e:
                # Debug: Print detailed error information for non-rate-limit errors
                error_str = str(e).lower()
                is_rate_limit = (
                    "rate limit" in error_str
                    or "429" in str(e)
                    or "rate_limit_exceeded" in error_str
                )
                
                if not is_rate_limit:
                    print(f"\n{'='*70}")
                    print(f"ERROR in generate_structured (phase: {phase})")
                    print(f"{'='*70}")
                    print(f"Model: {model}")
                    print(f"Output type: {output_type.__name__}")
                    print(f"Error type: {type(e).__name__}")
                    print(f"Error message: {e}")
                    if hasattr(e, "errors"):
                        import json
                        print("\nValidation errors:")
                        print(json.dumps(e.errors(), indent=2))
                    print(f"{'='*70}\n")
                raise

        # Run with retry logic using model-specific max_retries
        max_retries = self._get_max_retries(model_type)
        start_time = time.time()
        result = await self._retry_with_exponential_backoff(_run_agent, max_retries=max_retries)
        duration = time.time() - start_time

        # Track usage and cost
        usage = result.usage()
        self._track_usage(usage, model, phase, duration_seconds=duration)

        return result.output

    def generate_structured_sync(
        self,
        prompt: str,
        output_type: type[BaseModel],
        model_type: str = "default",
        phase: str = "general",
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Synchronous version of generate_structured."""
        model = self._select_model(model_type)

        # For structured output, allow validation retries
        agent = Agent(
            model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=3,  # Allow validation retries for structured output
        )
        
        # Create model settings with temperature
        model_settings = ModelSettings(temperature=self.temperature)

        start_time = time.time()
        result = agent.run_sync(prompt, model_settings=model_settings)
        duration = time.time() - start_time

        # Track usage and cost
        usage = result.usage()
        self._track_usage(usage, model, phase, duration_seconds=duration)

        return result.output

    async def generate_text(
        self,
        prompt: str,
        model_type: str = "default",
        phase: str = "general",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text output using pydantic-ai.

        Args:
            prompt: User prompt
            model_type: 'default', 'medium', or 'premium'
            phase: Phase name for cost tracking
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        model = self._select_model(model_type)

        # Apply model-specific rate limiting
        rate_limiter = self._get_rate_limiter(model_type)
        if rate_limiter:
            # Estimate tokens (prompt length + max_tokens or default)
            estimated_tokens = len(prompt.split()) * 1.3
            if max_tokens:
                estimated_tokens += max_tokens
            else:
                estimated_tokens += 1000  # Default estimate
            await rate_limiter.acquire(int(estimated_tokens))

        agent = Agent(
            model,
            system_prompt=system_prompt,
            retries=0,  # We handle retries ourselves
        )
        
        # Create model settings with model-specific temperature and max_tokens
        temperature = self._get_temperature(model_type)
        model_settings = ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async def _run_agent():
            return await agent.run(prompt, model_settings=model_settings)

        # Run with retry logic using model-specific max_retries
        max_retries = self._get_max_retries(model_type)
        start_time = time.time()
        result = await self._retry_with_exponential_backoff(_run_agent, max_retries=max_retries)
        duration = time.time() - start_time

        # Track usage and cost
        usage = result.usage()
        self._track_usage(usage, model, phase, duration_seconds=duration)

        return result.output

    def generate_text_sync(
        self,
        prompt: str,
        model_type: str = "default",
        phase: str = "general",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Synchronous version of generate_text."""
        model = self._select_model(model_type)

        agent = Agent(
            model,
            system_prompt=system_prompt,
            retries=self.max_retries,
        )
        
        # Create model settings with temperature and max_tokens
        model_settings = ModelSettings(
            temperature=self.temperature,
            max_tokens=max_tokens,
        )

        start_time = time.time()
        result = agent.run_sync(prompt, model_settings=model_settings)
        duration = time.time() - start_time

        # Track usage and cost
        usage = result.usage()
        self._track_usage(usage, model, phase, duration_seconds=duration)

        return result.output

    def get_cost_breakdown(self) -> CostBreakdown:
        """Get complete cost breakdown by phase."""
        total = sum(stats.cost_usd for stats in self.cost_tracker.values())
        return CostBreakdown(
            config_generation=self.cost_tracker["config_generation"],
            world_building=self.cost_tracker["world_building"],
            document_generation=self.cost_tracker["document_generation"],
            query_generation=self.cost_tracker["query_generation"],
            total_cost_usd=total,
        )

    def print_cost_summary(self) -> None:
        """Print a summary of costs."""
        breakdown = self.get_cost_breakdown()
        print("\n" + "=" * 50)
        print("Cost Summary")
        print("=" * 50)
        
        print(f"Config Generation:    ${breakdown.config_generation.cost_usd:.2f}")
        print(f"World Building:       ${breakdown.world_building.cost_usd:.2f}")
        print(f"Document Generation:  ${breakdown.document_generation.cost_usd:.2f}")
        print(f"Query Generation:     ${breakdown.query_generation.cost_usd:.2f}")
        
        print("-" * 50)
        print(f"TOTAL:                ${breakdown.total_cost_usd:.2f}")
        print("=" * 50 + "\n")

