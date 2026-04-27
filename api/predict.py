import os
import json
import re
import joblib
from http.server import BaseHTTPRequestHandler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LR_MODEL_PATH   = os.path.join(MODELS_DIR, "logistic_regression_ovr.joblib")

# ── Lazy-load models once on cold start ───────────────────────────────────────
_vectorizer = None
_model      = None

def _load_models():
    global _vectorizer, _model
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    if _model is None:
        _model = joblib.load(LR_MODEL_PATH)

# ── Text preprocessing (mirrors src/preprocess.py) ────────────────────────────
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\?!\.,]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Label metadata ─────────────────────────────────────────────────────────────
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

LABEL_META = {
    "toxic":         {"emoji": "☠️",  "color": "#ef4444"},
    "severe_toxic":  {"emoji": "💀",  "color": "#dc2626"},
    "obscene":       {"emoji": "🤬",  "color": "#f97316"},
    "threat":        {"emoji": "⚠️",  "color": "#eab308"},
    "insult":        {"emoji": "👊",  "color": "#a855f7"},
    "identity_hate": {"emoji": "🚫",  "color": "#ec4899"},
}

# ── Core prediction logic ──────────────────────────────────────────────────────
def run_prediction(text: str) -> dict:
    _load_models()
    cleaned  = clean_text(text)
    features = _vectorizer.transform([cleaned])
    flags    = _model.predict(features)[0]
    probs    = _model.predict_proba(features)[0]

    details = {}
    for i, label in enumerate(LABEL_COLS):
        details[label] = {
            "flag":        bool(flags[i]),
            "probability": round(float(probs[i]) * 100, 2),
            "emoji":       LABEL_META[label]["emoji"],
            "color":       LABEL_META[label]["color"],
        }

    is_safe       = not any(flags)
    overall_score = round(max(float(p) for p in probs) * 100, 2)

    return {
        "text":          text,
        "is_safe":       is_safe,
        "overall_score": overall_score,
        "details":       details,
    }

# ── Vercel handler ─────────────────────────────────────────────────────────────
class handler(BaseHTTPRequestHandler):

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json(200, {})

    def do_POST(self):
        try:
            length  = int(self.headers.get("Content-Length", 0))
            raw     = self.rfile.read(length)
            data    = json.loads(raw)
            text    = data.get("text", "").strip()

            if not text:
                self._send_json(400, {"error": "No text provided."})
                return

            if len(text) > 5000:
                self._send_json(400, {"error": "Text too long (max 5 000 chars)."})
                return

            result = run_prediction(text)
            self._send_json(200, result)

        except Exception as exc:
            self._send_json(500, {"error": str(exc)})
