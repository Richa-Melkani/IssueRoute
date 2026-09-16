from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path


# Create Flask application
app = Flask(__name__)


# Get project root directory
BASE_DIR = Path(__file__).resolve().parent


# Model path
MODEL_PATH = BASE_DIR / "models" / "issue_route_model.pkl"


# Load trained model
model = joblib.load(MODEL_PATH)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Complaint prediction API
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data from request
    data = request.get_json()

    # Get complaint text
    complaint = data.get("complaint", "").strip()

    # Check if complaint is empty
    if not complaint:
        return jsonify({
            "error": "Please enter a complaint."
        }), 400

    # Predict department
    prediction = model.predict([complaint])[0]

    # Get decision scores
    scores = model.decision_function([complaint])[0]

    # Get highest score
    max_score = max(scores)

    # Get second highest score
    sorted_scores = sorted(scores, reverse=True)
    second_highest_score = sorted_scores[1]

    # Calculate uncertainty margin
    margin = max_score - second_highest_score

    # Confidence threshold
    threshold = 0.3

    # Decide routing status
    if margin >= threshold:
        status = "Auto Routed"
    else:
        status = "Human Verification Required"

    # Return result
    return jsonify({
        "complaint": complaint,
        "predicted_department": prediction,
        "margin": round(float(margin), 4),
        "status": status
    })


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)