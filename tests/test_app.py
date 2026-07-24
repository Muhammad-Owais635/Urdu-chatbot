"""
Unit tests for the intent classifier and Flask API.

Run: pytest tests/
"""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("LLM_FALLBACK_ENABLED", "false")

from app import app as flask_app, classify  # noqa: E402


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# --- Classifier unit tests ---

def test_classify_returns_tuple():
    intent, confidence = classify("mera order kahan hai")
    assert isinstance(intent, str)
    assert isinstance(confidence, float)


def test_classify_order_status_roman_urdu():
    intent, _ = classify("mera order kahan hai")
    assert intent == "order_status"


def test_classify_greeting_urdu_script():
    intent, _ = classify("السلام علیکم")
    assert intent == "greeting"


def test_classify_gibberish_returns_unknown():
    intent, confidence = classify("asdkjaslkdjaslkdj qwerty zxcvbn")
    assert intent == "unknown"


# --- API endpoint tests ---

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_chat_endpoint_valid_message(client):
    res = client.post("/chat", json={"message": "mera order kahan hai"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["intent"] == "order_status"
    assert "response" in data
    assert "session_id" in data


def test_chat_endpoint_empty_message(client):
    res = client.post("/chat", json={"message": ""})
    assert res.status_code == 400


def test_chat_endpoint_missing_message_field(client):
    res = client.post("/chat", json={})
    assert res.status_code == 400


def test_chat_endpoint_message_too_long(client):
    long_message = "a" * 1000
    res = client.post("/chat", json={"message": long_message})
    assert res.status_code == 400


def test_chat_endpoint_session_persists(client):
    res1 = client.post("/chat", json={"message": "salam"})
    session_id = res1.get_json()["session_id"]

    res2 = client.post(
        "/chat", json={"message": "mera order kahan hai", "session_id": session_id}
    )
    assert res2.get_json()["session_id"] == session_id


def test_clear_session_endpoint(client):
    res1 = client.post("/chat", json={"message": "salam"})
    session_id = res1.get_json()["session_id"]

    res2 = client.delete(f"/session/{session_id}")
    assert res2.status_code == 200
    assert res2.get_json()["status"] == "cleared"


def test_404_returns_json(client):
    res = client.get("/this-route-does-not-exist")
    assert res.status_code == 404
    assert res.get_json()["error"] == "Not found"
