"""Generation configuration management."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelRateLimitConfig(BaseModel):
    """Rate limiting configuration for a specific model."""
    
    rpm: int | None = Field(
        default=None,
        description="Requests per minute (None=auto-detect)"
    )
    tpm: int | None = Field(
        default=None,
        description="Tokens per minute (None=auto-detect)"
    )
    max_retries: int | None = Field(
        default=None,
        description="Max retries on rate limit (None=auto-detect)"
    )


class ModelTierConfig(BaseModel):
    """Configuration for a model tier (default/medium/premium)."""
    
    model: str = Field(
        description="Model identifier in format 'provider:model-name'"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0=deterministic, 2=very creative)"
    )
    rate_limiting: ModelRateLimitConfig = Field(
        default_factory=ModelRateLimitConfig,
        description="Rate limiting settings for this model"
    )


class RateLimitConfig(BaseModel):
    """Global rate limiting configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable rate limiting globally"
    )
    initial_backoff: float = Field(
        default=1.0,
        description="Initial backoff time in seconds"
    )
    max_backoff: float = Field(
        default=60.0,
        description="Maximum backoff time in seconds"
    )


class ConcurrencyConfig(BaseModel):
    """Concurrency and parallelism configuration."""
    
    max_concurrent_generations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent document generations"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        description="Batch size for processing"
    )


class GenerationConfig(BaseModel):
    """Complete generation configuration."""
    
    default_model: ModelTierConfig = Field(
        default_factory=lambda: ModelTierConfig(
            model="gemini:gemini-2.0-flash-exp",
            temperature=0.7,
        ),
        description="Default model (fast, cheap)"
    )
    medium_model: ModelTierConfig = Field(
        default_factory=lambda: ModelTierConfig(
            model="openai:gpt-4o-mini",
            temperature=0.7,
        ),
        description="Medium quality model (balanced)"
    )
    premium_model: ModelTierConfig = Field(
        default_factory=lambda: ModelTierConfig(
            model="gemini:gemini-1.5-flash",
            temperature=0.7,
        ),
        description="Premium model (high quality, complex tasks)"
    )
    rate_limiting: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Global rate limiting settings"
    )
    concurrency: ConcurrencyConfig = Field(
        default_factory=ConcurrencyConfig,
        description="Concurrency settings"
    )
    
    @classmethod
    def load_from_file(cls, path: str | Path) -> "GenerationConfig":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            GenerationConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @classmethod
    def load_from_env(cls) -> "GenerationConfig":
        """
        Load configuration from environment variables (backward compatibility).
        
        Returns:
            GenerationConfig instance
        """
        config_dict: dict[str, Any] = {
            "default_model": {},
            "medium_model": {},
            "premium_model": {},
            "rate_limiting": {},
            "concurrency": {}
        }
        
        # Default model settings
        if default_model := os.getenv("DEFAULT_MODEL"):
            config_dict["default_model"]["model"] = default_model
        if temperature := os.getenv("TEMPERATURE"):
            config_dict["default_model"]["temperature"] = float(temperature)
        
        # Medium model settings
        if medium_model := os.getenv("MEDIUM_MODEL"):
            config_dict["medium_model"]["model"] = medium_model
        
        # Premium model settings
        if premium_model := os.getenv("PREMIUM_MODEL"):
            config_dict["premium_model"]["model"] = premium_model
        
        # Legacy rate limiting (applies to default model)
        if rpm := os.getenv("RATE_LIMIT_RPM"):
            if "rate_limiting" not in config_dict["default_model"]:
                config_dict["default_model"]["rate_limiting"] = {}
            config_dict["default_model"]["rate_limiting"]["rpm"] = int(rpm)
        if tpm := os.getenv("RATE_LIMIT_TPM"):
            if "rate_limiting" not in config_dict["default_model"]:
                config_dict["default_model"]["rate_limiting"] = {}
            config_dict["default_model"]["rate_limiting"]["tpm"] = int(tpm)
        if max_retries := os.getenv("MAX_RETRIES"):
            if "rate_limiting" not in config_dict["default_model"]:
                config_dict["default_model"]["rate_limiting"] = {}
            config_dict["default_model"]["rate_limiting"]["max_retries"] = int(max_retries)
        
        # Global rate limiting
        if disable_rate_limiting := os.getenv("DISABLE_RATE_LIMITING"):
            config_dict["rate_limiting"]["enabled"] = disable_rate_limiting.lower() != "true"
        
        # Concurrency
        if max_concurrent := os.getenv("MAX_CONCURRENT_GENERATIONS"):
            config_dict["concurrency"]["max_concurrent_generations"] = int(max_concurrent)
        if batch_size := os.getenv("BATCH_SIZE"):
            config_dict["concurrency"]["batch_size"] = int(batch_size)
        
        return cls(**config_dict)
    
    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "GenerationConfig":
        """
        Load configuration with fallback logic.
        
        Priority:
        1. Explicit config file path
        2. generation_config.yaml in current directory
        3. generation_config.yaml in workspace root
        4. Environment variables
        5. Defaults
        
        Args:
            config_path: Optional explicit path to config file
            
        Returns:
            GenerationConfig instance
        """
        # Try explicit path
        if config_path:
            return cls.load_from_file(config_path)
        
        # Try current directory
        if Path("generation_config.yaml").exists():
            return cls.load_from_file("generation_config.yaml")
        
        # Try workspace root (look for common indicators)
        for indicator in ["pyproject.toml", ".git", "README.md"]:
            current = Path.cwd()
            while current != current.parent:
                if (current / indicator).exists():
                    config_path = current / "generation_config.yaml"
                    if config_path.exists():
                        return cls.load_from_file(config_path)
                    break
                current = current.parent
        
        # Fall back to environment variables
        return cls.load_from_env()
    
    def save(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save config file
        """
        path = Path(path)
        with open(path, 'w') as f:
            yaml.safe_dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                sort_keys=False
            )
    
    def to_yaml(self) -> str:
        """
        Convert configuration to YAML string.
        
        Returns:
            YAML string representation
        """
        return yaml.safe_dump(
            self.model_dump(),
            default_flow_style=False,
            sort_keys=False
        )

