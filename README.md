# Urdu Customer Support Chatbot

A production-ready, bilingual (Roman Urdu + Urdu script) customer support chatbot with intent classification, session-based conversation memory, optional LLM-powered fallback, and a clean web chat UI — built with Flask and scikit-learn.

## Features

- **Bilingual understanding** — handles both Roman Urdu ("mera order kahan hai") and Urdu script ("میرا آرڈر کہاں ہے") in the same conversation
- **Intent classification** across 7 categories: greeting, order status, refund, complaint, pricing, human handoff, goodbye
- **Confidence-aware fallback** — low-confidence predictions are treated as "unknown" instead of guessing wrong
- **Optional LLM-powered fallback** — plug in an OpenAI (or compatible) API key to generate dynamic responses for out-of-scope messages; disabled by default so the bot works fully offline with zero API cost
- **Session-based conversation memory** — remembers recent turns per user session (in-memory, TTL-based expiry)
- **Production hardening**: rate limiting, input validation, structured logging (console + rotating file), consistent JSON error responses
- **Tested**: 12 automated tests covering the classifier and every API endpoint
- **Deployable**: Dockerfile + docker-compose included, gunicorn-ready
- **Upgrade path to BERT**: `bert_classifier.py` shows how to swap the lightweight classifier for a fine-tuned multilingual BERT model when you need higher accuracy on more complex phrasing

## Architecture

```
User message
     │
     ▼
Input validation (length, empty check)
     │
     ▼
Intent classifier (TF-IDF char n-grams + Logistic Regression)
     │
     ├─ confidence ≥ threshold → template response
     │
     └─ confidence < threshold ("unknown")
              │
              ├─ LLM fallback enabled → dynamic LLM response
              └─ LLM fallback disabled → static "unknown" template
     │
     ▼
Session store (conversation history, per session_id)
     │
     ▼
JSON response + logged
```

## Quick Start

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd urdu-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — defaults work out of the box)
cp .env.example .env

# 4. Train the intent classifier (takes a few seconds, no GPU needed)
python train_classifier.py

# 5. Run the app
python app.py
```

Open **http://localhost:5000** in your browser to chat, or call the API directly:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "mera order kahan hai"}'
```

## Running with Docker

```bash
docker compose up --build
```

This builds the image, trains the classifier at build time, and starts the server on port 5000.

## Running Tests

```bash
pytest tests/ -v
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web chat UI |
| `/chat` | POST | Send a message, get a classified + generated response |
| `/session/<session_id>` | DELETE | Clear a session's conversation history |
| `/health` | GET | Health check (model status, LLM fallback status) |

**POST /chat** request body:
```json
{
  "message": "mera order kahan hai",
  "session_id": "optional — omit to start a new session"
}
```

**Response:**
```json
{
  "session_id": "e569b1fd-5f67-4f6d-911e-0d9e174ea563",
  "message": "mera order kahan hai",
  "intent": "order_status",
  "confidence": 0.533,
  "response": "براہ کرم اپنا آرڈر نمبر بھیجیں...",
  "response_source": "template"
}
```

## Configuration

All settings are controlled via environment variables (see `.env.example`). Key ones:

| Variable | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum confidence to trust a classified intent |
| `LLM_FALLBACK_ENABLED` | `false` | Enable LLM-generated responses for out-of-scope messages |
| `OPENAI_API_KEY` | — | Required if LLM fallback is enabled |
| `RATE_LIMIT` | `30 per minute` | Per-IP rate limit on `/chat` |
| `MAX_HISTORY_TURNS` | `6` | How many conversation turns to remember per session |
| `MAX_MESSAGE_LENGTH` | `500` | Max characters accepted per message |

## Project Structure

```
urdu-chatbot/
├── app.py                  # Flask API + web UI server
├── config.py                # Centralized environment-based configuration
├── logging_config.py        # Logging setup (console + rotating file)
├── llm_fallback.py          # Optional LLM-powered fallback for unknown intents
├── session_store.py         # In-memory, TTL-based conversation history
├── train_classifier.py      # Trains the TF-IDF + Logistic Regression classifier
├── bert_classifier.py       # Optional: upgrade path to fine-tuned BERT
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── LICENSE
├── data/
│   ├── intents.json          # Training examples per intent
│   └── responses.json        # Response templates per intent
├── models/                   # Trained model saved here after training
├── templates/
│   └── index.html            # Chat UI
├── static/
│   ├── style.css
│   └── chat.js
└── tests/
    └── test_app.py           # 12 automated tests (classifier + API)
```

## How It Works

1. **Training** (`train_classifier.py`) loads example phrases per intent from `data/intents.json`, evaluates a TF-IDF (character n-gram) + Logistic Regression classifier on a held-out split for an honest accuracy report, then refits the final model on the **full** dataset before saving — so no training data goes to waste in the deployed model. Character n-grams (rather than word-level features) handle Roman Urdu spelling variation well, since there's no single "correct" spelling for most words.

2. **Inference** (`app.py`) classifies incoming messages and applies a confidence threshold to avoid confidently-wrong guesses on unfamiliar input. Known intents get a templated response; unknown ones optionally go to an LLM fallback.

3. **Conversation memory** (`session_store.py`) tracks recent turns per `session_id` so context can be passed to the LLM fallback, with automatic expiry after a period of inactivity.

4. **Upgrading accuracy**: `bert_classifier.py` shows how to fine-tune a multilingual BERT model on the same dataset for higher accuracy on more complex or ambiguous phrasing — useful once you've collected more real conversation data.

## Extending This Project

- Add more training examples to `data/intents.json` to cover more phrasings, then rerun `train_classifier.py`
- Add new intents by adding a key to both `intents.json` and `responses.json`
- Swap the in-memory `SessionStore` for Redis for multi-worker/multi-server deployments
- Enable LLM fallback for open-domain queries beyond the fixed intent set
- Add a `/feedback` endpoint to collect thumbs up/down on responses for future retraining

## Use Cases

- E-commerce customer support automation for Urdu-speaking markets
- FAQ automation for Pakistani / South Asian businesses
- Foundation for a full LLM-powered Urdu virtual assistant

## License

MIT — see [LICENSE](LICENSE).
