import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from groq import Groq

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
# Enable CORS so GitHub Pages frontend can call the Render backend
CORS(app)

# Read API configuration securely
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not api_key:
    raise RuntimeError(
        "GROQ_API_KEY is missing. Please configure it in your .env file."
    )

client = Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are MedPatent AI, an assistant that helps users explore
medical invention concepts and understand patent-related ideas.

Your responsibilities:
- Suggest possible medical invention concepts.
- Explain their potential purpose and technical features.
- Help users identify questions to investigate when researching patents.
- Explain that existing patents must be researched before claiming novelty.
- Never guarantee that an idea is novel or patentable.
- Do not present suggestions as proven medical advice or clinical guidance.
- Be clear, concise, and beginner-friendly.

Respond only to the user's current question.
Do not assume access to previous conversation messages.
"""


@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "service": "MedPatent AI API",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health"
        }
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    if not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "Please enter a question."}), 400

    user_message = user_message.strip()

    if len(user_message) > 4000:
        return jsonify(
            {"error": "Please keep your question under 4000 characters."}
        ), 400

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_completion_tokens=800,
        )

        answer = response.choices[0].message.content

        return jsonify({
            "answer": answer or "Sorry, I could not generate an answer."
        })

    except Exception:
        app.logger.exception("Groq API request failed.")
        return jsonify({
            "error": "Unable to generate a response right now. Please try again."
        }), 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
