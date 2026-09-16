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


# Load models
single_model = joblib.load(SINGLE_MODEL_PATH)

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
    data = request.get_json()

    # Get complaint text
    complaint = data.get("complaint", "").strip()


    # Check empty complaint
    if not complaint:
        return jsonify({
            "error": "Please enter a complaint."
        }), 400


    # -----------------------------------
    # STEP 1: Check for multiple issues
    # -----------------------------------

    # Simple rule-based indicators for multi-issue complaints
    multi_indicators = [
        " and ",
        "also",
        " as well as ",
        "along with",
        "plus"
    ]


    is_multi = any(
        indicator in complaint.lower()
        for indicator in multi_indicators
    )


    # -----------------------------------
    # MULTI-DEPARTMENT COMPLAINT
    # -----------------------------------

    if is_multi:

        prediction = multi_model.predict([complaint])

        departments = list(
            multi_mlb.inverse_transform(prediction)[0]
        )


        # If multiple departments detected
        if len(departments) >= 2:

            return jsonify({
                "complaint": complaint,
                "complaint_type": "Multi-Department",
                "departments": departments,
                "status": "Multiple Departments Detected"
            })


    # -----------------------------------
    # SINGLE-DEPARTMENT COMPLAINT
    # -----------------------------------

    prediction = single_model.predict([complaint])[0]


    # Decision scores
    scores = single_model.decision_function([complaint])[0]


    # Highest score
    max_score = max(scores)


    # Second highest score
    sorted_scores = sorted(scores, reverse=True)

    second_highest_score = sorted_scores[1]


    # Confidence margin
    margin = max_score - second_highest_score


    # Uncertainty threshold
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