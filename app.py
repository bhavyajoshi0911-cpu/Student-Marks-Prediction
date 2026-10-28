"""
app.py
======
Flask backend for the Student Marks Prediction System.
Provides routes for the dashboard, dataset overview, model performance,
about page, and the prediction API endpoint.
"""

import os
import json
import datetime
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"
METADATA_PATH = "model_metadata.json"

# Global variables for model and metadata
model = None
metadata = {}
dataset_df = None


def ensure_model_and_data():
    """Ensures dataset, model, and metadata exist. Trains them if missing."""
    global model, metadata, dataset_df
    
    if not os.path.exists(DATASET_PATH) or not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        print("[*] Required artifacts missing. Triggering training pipeline...")
        from train_model import train_and_evaluate
        train_and_evaluate()

    # Load dataset
    dataset_df = pd.read_csv(DATASET_PATH)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("[*] Model, dataset, and metadata successfully loaded into memory.")


# Load immediately on module import
ensure_model_and_data()


@app.context_processor
def inject_global_vars():
    """Provides dynamic variables to all templates."""
    now = datetime.datetime.now()
    # Student friendly time-of-day greeting
    hour = now.hour
    if hour < 12:
        greeting = "Good morning, Student! ☀️"
    elif hour < 17:
        greeting = "Good afternoon, Student! 🎓"
    else:
        greeting = "Good evening, Student! 🌙"

    return {
        "current_year": now.year,
        "student_greeting": greeting,
        "project_name": "Student Marks Prediction System"
    }


def calculate_grade(marks):
    """Returns grade and student performance classification based on marks."""
    if marks >= 90:
        return "A+ (Outstanding)", "success", "🌟 Exceptional dedication! Keep up the brilliant standard."
    elif marks >= 80:
        return "A (Excellent)", "success", "🎯 Excellent preparation! You are positioned for top honors."
    elif marks >= 70:
        return "B+ (Very Good)", "primary", "📈 Very solid performance. A little extra revision can push you to an A."
    elif marks >= 60:
        return "B (Good)", "info", "👍 Decent grasp of concepts. Consistent daily review will boost your score."
    elif marks >= 50:
        return "C (Average)", "warning", "⚠️ Passing range. Increase daily study hours to improve confidence."
    elif marks >= 40:
        return "D (Pass)", "warning", "⚡ Borderline passing. Dedicated study routines are highly recommended."
    else:
        return "F (Needs Significant Improvement)", "danger", "🚨 High risk of failing. Consider seeking extra academic guidance."


@app.route("/")
def index():
    """Main Dashboard Route."""
    # Ensure fresh read of preview and stats
    preview_rows = dataset_df.head(5).to_dict(orient="records")
    
    return render_template(
        "index.html",
        active_page="dashboard",
        stats=metadata.get("dataset_stats", {}),
        metrics=metadata.get("metrics", {}),
        equation=metadata.get("equation", ""),
        slope=metadata.get("slope", 0.0),
        intercept=metadata.get("intercept", 0.0),
        preview_rows=preview_rows,
        scatter_data=metadata.get("scatter_data", []),
        regression_line=metadata.get("regression_line", {})
    )


@app.route("/data-overview")
def data_overview():
    """Data Overview Page showing full dataset, statistics, and distributions."""
    records = dataset_df.to_dict(orient="records")
    
    # Statistical Summary calculations
    hours_desc = dataset_df["Study_Hours"].describe().to_dict()
    marks_desc = dataset_df["Marks"].describe().to_dict()

    summary_stats = [
        {"metric": "Count (Records)", "hours": int(hours_desc["count"]), "marks": int(marks_desc["count"])},
        {"metric": "Mean (Average)", "hours": round(hours_desc["mean"], 2), "marks": round(marks_desc["mean"], 2)},
        {"metric": "Standard Deviation", "hours": round(hours_desc["std"], 2), "marks": round(marks_desc["std"], 2)},
        {"metric": "Minimum", "hours": round(hours_desc["min"], 2), "marks": round(marks_desc["min"], 2)},
        {"metric": "25th Percentile (Q1)", "hours": round(hours_desc["25%"], 2), "marks": round(marks_desc["25%"], 2)},
        {"metric": "50th Percentile (Median)", "hours": round(hours_desc["50%"], 2), "marks": round(marks_desc["50%"], 2)},
        {"metric": "75th Percentile (Q3)", "hours": round(hours_desc["75%"], 2), "marks": round(marks_desc["75%"], 2)},
        {"metric": "Maximum", "hours": round(hours_desc["max"], 2), "marks": round(marks_desc["max"], 2)},
    ]

    return render_template(
        "data_overview.html",
        active_page="data_overview",
        records=records,
        summary_stats=summary_stats,
        total_records=len(dataset_df)
    )


@app.route("/model-performance")
def model_performance():
    """Model Performance & Evaluation Page."""
    return render_template(
        "model_performance.html",
        active_page="model_performance",
        metrics=metadata.get("metrics", {}),
        equation=metadata.get("equation", ""),
        slope=metadata.get("slope", 0.0),
        intercept=metadata.get("intercept", 0.0),
        test_comparison=metadata.get("test_comparison", []),
        stats=metadata.get("dataset_stats", {})
    )


@app.route("/about")
def about():
    """About Project Page."""
    return render_template(
        "about.html",
        active_page="about",
        metrics=metadata.get("metrics", {}),
        stats=metadata.get("dataset_stats", {})
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Prediction Endpoint.
    Supports JSON POST requests from frontend JavaScript fetch,
    as well as standard HTML Form POST requests and GET requests.
    """
    if request.method == "GET":
        # Redirect to dashboard prediction card or render view
        return redirect(url_for("index") + "#predict-panel")

    # Determine input data format (JSON vs Form-Data)
    hours_input = None
    is_json = request.is_json
    
    if is_json:
        data = request.get_json() or {}
        hours_input = data.get("study_hours")
    else:
        hours_input = request.form.get("study_hours")

    # 1. Validation: Empty Check
    if hours_input is None or str(hours_input).strip() == "":
        err_msg = "Please enter the number of study hours."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("index.html", error_message=err_msg, active_page="dashboard")

    # 2. Validation: Numeric Check
    try:
        hours = float(hours_input)
    except (ValueError, TypeError):
        err_msg = "Invalid input. Study hours must be a valid number (e.g. 5.5)."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("index.html", error_message=err_msg, active_page="dashboard")

    # 3. Validation: Non-negative Check
    if hours < 0:
        err_msg = "Study hours cannot be negative. Please enter a value of 0 or greater."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("index.html", error_message=err_msg, active_page="dashboard")

    # 4. Validation: Realistic Daily Boundary Check (<= 24 hours per day)
    if hours > 24:
        err_msg = "Study hours cannot exceed 24 hours in a single day. Please enter a realistic number."
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 400
        return render_template("index.html", error_message=err_msg, active_page="dashboard")

    # Perform Prediction using Scikit-Learn Model
    try:
        input_df = pd.DataFrame({"Study_Hours": [hours]})
        raw_prediction = float(model.predict(input_df)[0])
        
        # Real-world exam marks are bounded between 0 and 100
        bounded_marks = round(min(100.0, max(0.0, raw_prediction)), 1)
        raw_marks = round(raw_prediction, 1)

        grade, grade_class, advice = calculate_grade(bounded_marks)

        result_payload = {
            "success": True,
            "study_hours": round(hours, 2),
            "predicted_marks": bounded_marks,
            "raw_marks": raw_marks,
            "grade": grade,
            "grade_class": grade_class,
            "advice": advice,
            "is_capped": bounded_marks != raw_marks,
            "equation": metadata.get("equation", ""),
            "r2_score": metadata.get("metrics", {}).get("r2_score", 0.0),
            "message": f"Predicted Marks: {bounded_marks}/100"
        }

        if is_json:
            return jsonify(result_payload), 200
        else:
            return render_template(
                "index.html",
                active_page="dashboard",
                prediction_result=result_payload,
                stats=metadata.get("dataset_stats", {}),
                metrics=metadata.get("metrics", {}),
                preview_rows=dataset_df.head(5).to_dict(orient="records"),
                scatter_data=metadata.get("scatter_data", []),
                regression_line=metadata.get("regression_line", {})
            )

    except Exception as e:
        err_msg = f"An unexpected error occurred during prediction: {str(e)}"
        if is_json:
            return jsonify({"success": False, "error": err_msg}), 500
        return render_template("index.html", error_message=err_msg, active_page="dashboard")


@app.route("/api/chart-data")
def api_chart_data():
    """API endpoint providing dataset scatter points and linear fit line for Chart.js."""
    return jsonify({
        "scatter_data": metadata.get("scatter_data", []),
        "regression_line": metadata.get("regression_line", {}),
        "equation": metadata.get("equation", "")
    })


@app.route("/download-dataset")
def download_dataset():
    """Allows downloading the dataset CSV file."""
    if os.path.exists(DATASET_PATH):
        return send_file(DATASET_PATH, as_attachment=True, download_name="student_marks_dataset.csv")
    return jsonify({"error": "Dataset file not found."}), 404


@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 handler."""
    return render_template("404.html"), 404


if __name__ == "__main__":
    # Run development server
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
