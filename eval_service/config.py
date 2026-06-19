"""
config.py — Central configuration for the Vidya eval service.

All environment variables are declared here. Import Settings from this module
everywhere — never read os.environ directly in application code.

Usage:
    from eval_service.config import settings

    url = settings.supabase_url
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini ────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_classifier_api_key: str = ""
    gemini_frontier_api_key: str = ""

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_service_role_key: str = ""

    # ── AI service (Himanshu — for SK-08 live eval) ───────────────────────────
    ai_service_url: str = "http://127.0.0.1:8000"
    ai_service_token: str = ""
    ai_user_token: str = "eval-harness"

    # ── Retrieval (locked from SK-06 tuning) ──────────────────────────────────
    retrieval_match_count: int = 3
    retrieval_min_similarity: float = 0.0
    retrieval_ef_search: int = 40

    # ── Mastery (FSRS-4.5) ────────────────────────────────────────────────────
    mastery_desired_retention: float = 0.90

    # ── Eval flags ────────────────────────────────────────────────────────────
    sk08_live: bool = False           # True → use live AI service for SK-08
    sk08_mock_accuracy: float = 0.93  # mock accuracy when not live
    sk08_dataset_path: str = ""       # override CSV path (defaults to ci_scenarios.csv in CI)

    sk10_live: bool = False           # True → use live RT-18 KB edge function
    sk10_direct: bool = False         # True → embed via Gemini + call RPC directly (CI mode)
    sk10_mock_mode: str = "isolated"  # "isolated" | "leaky"

    # ── Service ───────────────────────────────────────────────────────────────
    environment: str = "dev"          # "dev" | "staging" | "prod"
    port: int = 8002
    log_level: str = "info"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
