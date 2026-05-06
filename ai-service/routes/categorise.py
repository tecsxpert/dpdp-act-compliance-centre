import json
import time
from flask import Blueprint, request, jsonify
from services.groq_client import GroqClient

categorise_bp = Blueprint("categorise", __name__)
client = GroqClient()

# Predefined DPDP Act categories
DPDP_CATEGORIES = [
    "Data Collection",
    "Data Storage",
    "Data Processing",
    "Consent Management",
    "Data Breach",
    "Third-Party Sharing",
    "Data Retention",
    "User Rights"
]

FALLBACK_RESPONSE = {
    "category": "Data Processing",
    "confidence": 0.0,
    "reasoning": "AI service temporarily unavailable. Please retry.",
    "meta": {
        "model_used": "llama-3.3-70b-versatile",
        "tokens_used": 0,
        "response_time_ms": 0,
        "cached": False,
        "is_fallback": True
    }
}


@categorise_bp.route("/categorise", methods=["POST"])
def categorise():
    """
    POST /categorise
    Body: { "text": "description of compliance item" }
    Returns: { category, confidence, reasoning, meta }
    """
    data = request.get_json()

    # ── Input validation ──────────────────────────────────────────────────────
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "'text' field cannot be empty"}), 400
    if len(text) > 5000:
        return jsonify({"error": "'text' too long. Max 5000 characters."}), 400

    categories_str = " | ".join(DPDP_CATEGORIES)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a DPDP Act (Digital Personal Data Protection Act) compliance expert. "
                "Respond ONLY in valid JSON. No extra text, no markdown fences."
            )
        },
        {
            "role": "user",
            "content": (
                f"Classify this compliance item into ONE of these categories:\n"
                f"{categories_str}\n\n"
                f"Compliance Item: {text}\n\n"
                f"Respond with ONLY this JSON:\n"
                f'{{"category": "one of the categories above", '
                f'"confidence": 0.95, '
                f'"reasoning": "one clear sentence explaining the classification"}}'
            )
        }
    ]

    start = time.time()
    result = client.call_for_json(messages, temperature=0.1)
    elapsed_ms = int((time.time() - start) * 1000)

    # ── Fallback if AI failed ─────────────────────────────────────────────────
    if result is None:
        return jsonify(FALLBACK_RESPONSE), 200

    # ── Validate AI returned a known category ─────────────────────────────────
    if result.get("category") not in DPDP_CATEGORIES:
        result["category"] = "Data Processing"   # safe default

    # ── Clamp confidence between 0.0 and 1.0 ─────────────────────────────────
    confidence = float(result.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return jsonify({
        "category":  result.get("category"),
        "confidence": confidence,
        "reasoning":  result.get("reasoning", ""),
        "meta": {
            "model_used":       "llama-3.3-70b-versatile",
            "tokens_used":      0,
            "response_time_ms": elapsed_ms,
            "cached":           False,
            "is_fallback":      False
        }
    }), 200
