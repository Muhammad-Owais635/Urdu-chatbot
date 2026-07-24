"""
Urdu Customer Support Chatbot — Production Flask API + Web UI

Features:
- Intent classification (Roman Urdu + Urdu script)
- Session-based conversation memory
- Optional LLM-powered fallback for out-of-scope messages
- Rate limiting, input validation, structured logging
- Proper JSON error responses for all failure modes

Run:
    python train_classifier.py     # once, to produce models/intent_classifier.joblib
    python app.py

Then open http://localhost:5000
"""

import logging
import os
import uuid

import joblib
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import HTTPException

from config import Config
from logging_config import setup_logging
from llm_fallback import LLMFallback
from session_store import SessionStore

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# --- App & config ---
app = Flask(__name__)
app.config.from_object(Config)

logger = setup_logging(Config)
CORS(app, origins=Config.ALLOWED_ORIGINS)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[Config.RATE_LIMIT],
    storage_uri="memory://",
)

# --- Load intent classifier ---
model_path = os.path.join(APP_DIR, Config.MODEL_PATH)
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Trained model not found at {model_path}. "
        "Run `python train_classifier.py` first."
    )
classifier = joblib.load(model_path)
logger.info("Intent classifier loaded from %s", model_path)

# --- Load response templates ---
import json
responses_path = os.path.join(APP_DIR, "data", "responses.json")
with open(responses_path, "r", encoding="utf-8") as f:
    RESPONSES = json.load(f)

# --- Session memory + LLM fallback ---
session_store = SessionStore(
    max_turns=Config.MAX_HISTORY_TURNS, ttl_seconds=Config.SESSION_TTL_SECONDS
)
llm_fallback = LLMFallback(Config)
if llm_fallback.enabled:
    logger.info("LLM fallback enabled using model %s", Config.LLM_MODEL)
else:
    logger.info("LLM fallback disabled — using template responses only")


def classify(text: str):
    """Returns (intent, confidence) for the given input text."""
    probs = classifier.predict_proba([text])[0]
    classes = classifier.classes_
    best_idx = probs.argmax()
    intent = classes[best_idx]
    confidence = float(probs[best_idx])

    if confidence < Config.CONFIDENCE_THRESHOLD:
        return "unknown", confidence
    return intent, confidence


def validate_message(message):
    if not message or not message.strip():
        return "Message cannot be empty."
    if len(message) > Config.MAX_MESSAGE_LENGTH:
        return f"Message too long (max {Config.MAX_MESSAGE_LENGTH} characters)."
    return None


# --- Routes ---

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
@limiter.limit(Config.RATE_LIMIT)
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    error = validate_message(message)
    if error:
        return jsonify({"error": error}), 400

    intent, confidence = classify(message)
    history = session_store.get_history(session_id)

    if intent == "unknown":
        llm_response = llm_fallback.generate_fallback_response(message, history)
        response_text = llm_response or RESPONSES["unknown"]
        source = "llm" if llm_response else "template"
    else:
        response_text = RESPONSES.get(intent, RESPONSES["unknown"])
        source = "template"

    session_store.add_turn(session_id, "user", message)
    session_store.add_turn(session_id, "assistant", response_text)

    logger.info(
        "session=%s intent=%s confidence=%.3f source=%s",
        session_id, intent, confidence, source,
    )

    return jsonify({
        "session_id": session_id,
        "message": message,
        "intent": intent,
        "confidence": round(confidence, 3),
        "response": response_text,
        "response_source": source,
    })


@app.route("/session/<session_id>", methods=["DELETE"])
def clear_session(session_id):
    session_store.clear(session_id)
    return jsonify({"status": "cleared", "session_id": session_id})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": classifier is not None,
        "llm_fallback_enabled": llm_fallback.enabled,
    })


# --- Error handlers ---

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "detail": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Too many requests. Please slow down."}), 429


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.name, "detail": e.description}), e.code


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    logger.exception("Unhandled exception")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)
