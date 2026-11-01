from flask import Flask, render_template, request, jsonify
import joblib
from pathlib import Path


# Create Flask application
app = Flask(__name__)


# Project directory
BASE_DIR = Path(__file__).resolve().parent


# Model paths
SINGLE_MODEL_PATH = BASE_DIR / "models" / "issue_route_model.pkl"
MULTI_MODEL_PATH = BASE_DIR / "models" / "multi_department_model.pkl"
COMPLAINT_TYPE_MODEL_PATH = BASE_DIR / "models" / "complaint_type_model.pkl"


# Load models
single_model = joblib.load(SINGLE_MODEL_PATH)
complaint_type_model = joblib.load(COMPLAINT_TYPE_MODEL_PATH)

multi_model_data = joblib.load(MULTI_MODEL_PATH)
multi_model = multi_model_data["model"]
multi_mlb = multi_model_data["mlb"]


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Complaint prediction
@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON data
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Invalid request. Please send valid JSON data."
        }), 400

    # Get and validate complaint text
    complaint = data.get("complaint", "")

    if not isinstance(complaint, str):
        return jsonify({
            "error": "Complaint must be text."
        }), 400

    complaint = complaint.strip()

    if not complaint:
        return jsonify({
            "error": "Please enter a complaint."
        }), 400

    # -----------------------------------
    # STEP 1: Predict complaint type
    # -----------------------------------

    complaint_type = complaint_type_model.predict([complaint])[0]
    is_multi = complaint_type == "multi"

    # Handle unknown complaints
    if complaint_type == "unknown":
        return jsonify({
            "complaint": complaint,
            "complaint_type": "Unknown",
            "departments": [],
            "status": "Human Verification Required",
            "message": "The complaint could not be classified confidently. Manual review is required."
        })

    # -----------------------------------
    # STEP 2: Multi-department complaint
    # -----------------------------------

    if is_multi:

        prediction = multi_model.predict([complaint])

        departments = list(
            multi_mlb.inverse_transform(prediction)[0]
        )

        if len(departments) >= 2:
            return jsonify({
                "complaint": complaint,
                "complaint_type": "Multi-Department",
                "departments": departments,
                "status": "Multiple Departments Detected"
            })

        return jsonify({
            "complaint": complaint,
            "complaint_type": "Multi-Department",
            "departments": departments,
            "status": "Human Verification Required",
            "message": "Multiple issues were detected, but department routing needs manual verification."
        })

    # -----------------------------------
    # STEP 3: Single-department complaint
    # -----------------------------------

    prediction = single_model.predict([complaint])[0]

    # Get decision scores
    scores = single_model.decision_function([complaint])[0]

    # Calculate confidence margin
    sorted_scores = sorted(scores, reverse=True)

    if len(sorted_scores) < 2:
        return jsonify({
            "complaint": complaint,
            "complaint_type": "Single-Department",
            "departments": [prediction],
            "status": "Human Verification Required",
            "message": "Insufficient prediction scores. Manual review is required."
        })

    margin = sorted_scores[0] - sorted_scores[1]

    # Uncertainty threshold (demo threshold; requires evaluation)
    threshold = 0.3

    if margin >= threshold:
        status = "Auto Routed"
    else:
        status = "Human Verification Required"

    return jsonify({
        "complaint": complaint,
        "complaint_type": "Single-Department",
        "departments": [prediction],
        "margin": round(float(margin), 4),
        "status": status
    })


# Run application
if __name__ == "__main__":
    app.run(debug=True)
