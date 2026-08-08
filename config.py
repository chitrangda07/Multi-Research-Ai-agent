"""Centralized configuration, environment validation, and logging setup."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    tavily_api_key: str
    model_name: str
    temperature: float
    request_timeout: int
    max_search_results: int
    scrape_char_limit: int
    snippet_char_limit: int


def get_settings() -> Settings:
    """Read and validate configuration from environment variables.

    Raises:
        ConfigError: if a required API key is missing.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    missing = [
        name
        for name, val in (("OPENAI_API_KEY", openai_key), ("TAVILY_API_KEY", tavily_key))
        if not val
    ]
    if missing:
        raise ConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your keys."
        )

    return Settings(
        openai_api_key=openai_key,  # type: ignore[arg-type]
        tavily_api_key=tavily_key,  # type: ignore[arg-type]
        model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
        max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "5")),
        scrape_char_limit=int(os.getenv("SCRAPE_CHAR_LIMIT", "6000")),
        snippet_char_limit=int(os.getenv("SNIPPET_CHAR_LIMIT", "400")),
    )