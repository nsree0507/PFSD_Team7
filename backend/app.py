import sys
import os
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from db_connection import users_collection, tickets_collection
from classifier.predict import enhanced_prediction  # use your existing model
from classifier.model import load_model
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)

classifier = load_model()

# ---------------- HOME ----------------
@app.route("/")
def home():
    return send_file("../smart-helpdesk-dashboard.html")

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json

    if not data.get("email") or not data.get("password"):
        return jsonify({"error": "Email and password required"}), 400

    user = {
        "name": data.get("name", ""),
        "email": data["email"],
        "password": data["password"],
        "created_at": datetime.now()
    }

    users_collection.insert_one(user)

    return jsonify({"message": "User registered successfully"})

# ---------------- LOGIN ----------------
@app.route("/signin", methods=["POST"])
def signin():
    data = request.json

    user = users_collection.find_one({
        "email": data["email"],
        "password": data["password"]
    })

    if user:
        user_id = str(user["_id"])

        return jsonify({
            "message": "Login successful",
            "user_id": user_id,   # ✅ ADD THIS
            "name": user.get("name", ""),
            "email": user.get("email", "")
        })

    return jsonify({"message": "Invalid credentials"}), 401
# ---------------- CREATE TICKET ----------------
@app.route("/tickets", methods=["POST"])
def create_ticket():
    data = request.json

    # ---------------- VALIDATION ----------------
    if not data.get("user_id") or not data.get("query", "").strip():
        return jsonify({"error": "user_id and query required"}), 400

    labels = [
        "Fees Issue",
        "Attendance Issue",
        "Marks Issue",
        "Exam Query",
        "Hostel Issue",
        "General Query"
    ]

    # ---------------- ML PREDICTION ----------------
    result = enhanced_prediction(data["query"], labels)

    # ---------------- SAFE EXTRACTION ----------------
    category = (
        result.get("predicted_label")
        or result.get("label")
        or (result.get("labels", [None])[0])
        or "Unknown"
    )

    confidence = (
        result.get("confidence")
        or (result.get("scores", [None])[0])
        or 0
    )

    confidence = float(confidence) if confidence else 0.0

    # ---------------- DB OBJECT ----------------
    ticket = {
        "user_id": data["user_id"],
        "query": data["query"],
        "category": category,
        "confidence": confidence,
        "status": "Open",
        "created_at": datetime.now(timezone.utc)
    }

    tickets_collection.insert_one(ticket)

    return jsonify({
        "message": "Ticket created",
        "category": category,
        "confidence": confidence
    })

# ---------------- GET TICKETS ----------------
@app.route("/tickets/<user_id>", methods=["GET"])
def get_tickets(user_id):
    tickets = list(tickets_collection.find(
        {"user_id": user_id},
        {"_id": 0}
    ))
    return jsonify(tickets)

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json()
        query = data.get("query", "")

        result = enhanced_prediction(query)
        print(result)
        return jsonify(result)

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": str(e)}), 500

def get_current_user():
    return request.json.get("user_id")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
