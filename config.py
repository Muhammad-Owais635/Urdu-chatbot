"""
Centralized application configuration.

All environment-specific settings live here and are loaded from environment
variables (via a .env file in development, or real env vars in production).
Never hardcode secrets — copy .env.example to .env and fill in your own values.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me-in-production")
    DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # --- Intent classifier ---
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.35))
    MODEL_PATH = os.getenv("MODEL_PATH", "models/intent_classifier.joblib")

    # --- LLM fallback (optional — used for queries outside known intents) ---
    LLM_FALLBACK_ENABLED = os.getenv("LLM_FALLBACK_ENABLED", "false").lower() == "true"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_SYSTEM_PROMPT = os.getenv(
        "LLM_SYSTEM_PROMPT",
        "You are a helpful, polite customer support assistant. Reply in the "
        "same language/script the user wrote in (Roman Urdu or Urdu script). "
        "Keep responses short and clear."
    )

    # --- Rate limiting ---
    RATE_LIMIT = os.getenv("RATE_LIMIT", "30 per minute")

    # --- Conversation memory ---
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 6))
    SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 1800))  # 30 min

    # --- Input validation ---
    MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 500))

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

    # --- CORS ---
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
