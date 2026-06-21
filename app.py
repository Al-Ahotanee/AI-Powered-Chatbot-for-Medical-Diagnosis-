import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are Aegis, an AI medical triage assistant designed to help users understand their symptoms before they see a physician.

Your responsibilities:
1. Listen carefully to reported symptoms and ask targeted clarifying questions (onset, severity 1-10, location, duration, aggravating/relieving factors).
2. Identify the most probable conditions and explain them clearly in plain language.
3. Triage urgency into one of three explicit tiers:
   - 🔴 EMERGENCY: Advise calling 911 or going to an ER immediately.
   - 🟡 SEE A DOCTOR: Recommend scheduling an appointment within 24–72 hours.
   - 🟢 HOME CARE: Provide safe, general self-care guidance.
4. Always remind the user you are an AI and cannot replace a licensed physician.
5. Never name specific prescription drugs or dosages. You may reference general classes (e.g., "OTC antihistamines").
6. Use empathetic, calm, precise language. Avoid jargon without explanation.
7. If the user expresses distress or describes emergency symptoms (severe chest pain, stroke symptoms, difficulty breathing, suicidal ideation), lead with the emergency directive immediately.

Format responses with clear structure using markdown-style bold for key terms when helpful. Keep responses concise but complete."""


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not configured")
        return jsonify({"error": "Server configuration error: AI service not configured."}), 500

    data = request.get_json(silent=True)
    if not data or not data.get("history"):
        return jsonify({"error": "Request body must include a 'history' array."}), 400

    front_history: list[dict] = data["history"]

    # Validate history shape
    for msg in front_history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return jsonify({"error": "Each history item must have 'role' and 'content' fields."}), 400

    # Build Gemini history (exclude the last user message and system messages)
    gemini_history: list[types.Content] = []
    for msg in front_history[:-1]:
        if msg["role"] in ("system", "error"):
            continue
        role = "model" if msg["role"] == "assistant" else "user"
        gemini_history.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )

    new_user_message = front_history[-1].get("content", "").strip()
    if not new_user_message:
        return jsonify({"error": "Last message content cannot be empty."}), 400

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1024,
            temperature=0.4,
        )
        chat_session = client.chats.create(
            model="gemini-2.5-flash-lite",
            config=config,
            history=gemini_history,
        )
        response = chat_session.send_message(new_user_message)
        logger.info("Gemini response generated successfully.")
        return jsonify({"text": response.text})

    except Exception as exc:
        err = str(exc)
        logger.error("Gemini API error: %s", err)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return jsonify({"error": "AI service rate limit reached. Please try again in a moment."}), 429
        if "403" in err or "PERMISSION_DENIED" in err:
            return jsonify({"error": "AI service authentication failed. Check the API key."}), 403
        return jsonify({"error": "AI service temporarily unavailable. Please try again."}), 502


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed."}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
