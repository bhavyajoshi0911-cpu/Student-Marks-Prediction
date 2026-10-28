"""
train_model.py
==============
End-to-end Machine Learning pipeline for Student Marks Prediction.
- Loads and inspects dataset.csv (creates realistic dataset if missing)
- Cleans and validates data (types, duplicates, range checks)
- Splits data into training and test sets (train_test_split)
- Trains LinearRegression model using Scikit-Learn
- Evaluates model using MAE, MSE, RMSE, and R2 Score
- Saves trained model to model.pkl using Joblib
- Saves comprehensive model metadata to model_metadata.json
- Generates high-resolution Matplotlib visualizations saved to static/images/
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASET_PATH = "dataset.csv"
MODEL_PATH = "model.pkl"
METADATA_PATH = "model_metadata.json"
IMAGES_DIR = os.path.join("static", "images")


def create_realistic_dataset(filepath=DATASET_PATH, n_samples=100, random_state=42):
    """Generates a realistic student study hours vs exam marks dataset."""
    np.random.seed(random_state)
    
    # Study hours realistically distributed between 1.0 and 10.5 hours
    raw_hours = np.random.uniform(1.0, 10.0, n_samples) + np.random.normal(0, 0.4, n_samples)
    study_hours = np.round(np.clip(raw_hours, 0.5, 12.0), 1)
    
    # Marks: base score ~16.5 + 7.8 marks/hour with realistic student performance variance
    noise = np.random.normal(0, 4.5, n_samples)
    raw_marks = 16.5 + (7.8 * study_hours) + noise
    marks = np.round(np.clip(raw_marks, 10.0, 98.5), 1)
    
    df = pd.DataFrame({
        "ID": range(1, n_samples + 1),
        "Study_Hours": study_hours,
        "Marks": marks
    })
    
    df.to_csv(filepath, index=False)
    print(f"[*] Generated synthetic dataset with {n_samples} records at: {filepath}")
    return df


def load_and_validate_dataset(filepath=DATASET_PATH):
    """Loads CSV dataset and performs validation & cleaning checks."""
    if not os.path.exists(filepath):
        print(f"[!] {filepath} not found. Creating a new realistic dataset...")
        create_realistic_dataset(filepath)

    df = pd.read_csv(filepath)
    print("=" * 60)
    print(f"[*] Loading dataset from: {filepath}")
    print(f"[*] Raw Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[*] Column Names: {list(df.columns)}")

    # Standardize column naming if variations exist
    col_map = {}
    for col in df.columns:
        norm = col.strip().lower().replace(" ", "_")
        if "hour" in norm:
            col_map[col] = "Study_Hours"
        elif "mark" in norm or "score" in norm:
            col_map[col] = "Marks"
        elif "id" in norm:
            col_map[col] = "ID"
    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if "Study_Hours" not in df.columns or "Marks" not in df.columns:
        raise ValueError("Dataset must contain 'Study_Hours' and 'Marks' columns.")

    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))

    # Inspect missing values
    missing_counts = df.isnull().sum()
    print("[*] Missing Values Check:")
    for col, count in missing_counts.items():
        print(f"    - {col}: {count} missing")

    # Drop missing values if any
    initial_len = len(df)
    df = df.dropna(subset=["Study_Hours", "Marks"]).copy()
    if len(df) < initial_len:
        print(f"[!] Dropped {initial_len - len(df)} rows with missing values.")

    # Convert to numeric
    df["Study_Hours"] = pd.to_numeric(df["Study_Hours"], errors="coerce")
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
    df = df.dropna(subset=["Study_Hours", "Marks"]).copy()

    # Range validation: hours >= 0, 0 <= marks <= 100
    valid_mask = (df["Study_Hours"] >= 0) & (df["Marks"] >= 0) & (df["Marks"] <= 100)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        print(f"[!] Removing {invalid_count} records violating boundaries (hours >= 0, 0 <= marks <= 100).")
        df = df[valid_mask].copy()

    # Re-index ID if needed
    df["ID"] = range(1, len(df) + 1)
    
    print(f"[*] Cleaned Dataset Shape: {df.shape[0]} valid rows")
    print("=" * 60)
    return df


def generate_visualizations(df, model, X_test, y_test, y_pred):
    """Generates professional Matplotlib visualizations and saves them to static/images."""
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # 1. Study Hours vs Marks Scatter + Regression Line
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Scatter points
    plt.scatter(
        df["Study_Hours"], 
        df["Marks"], 
        color="#2878F0", 
        alpha=0.75, 
        edgecolors="#142B50", 
        linewidths=0.7,
        s=60, 
        label="Actual Student Records"
    )

    # Regression Line
    hours_range = pd.DataFrame({"Study_Hours": np.linspace(df["Study_Hours"].min(), df["Study_Hours"].max(), 200)})
    marks_line = model.predict(hours_range)
    plt.plot(
        hours_range["Study_Hours"], 
        marks_line, 
        color="#10B981", 
        linewidth=2.5, 
        label=f"Linear Fit: Marks = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}"
    )

    plt.title("Study Hours vs Marks (Linear Regression)", fontsize=14, fontweight="bold", color="#172B4D", pad=15)
    plt.xlabel("Study Hours (per day)", fontsize=11, fontweight="600", color="#172B4D")
    plt.ylabel("Exam Marks (out of 100)", fontsize=11, fontweight="600", color="#172B4D")
    plt.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", loc="upper left")
    plt.tight_layout()
    chart1_path = os.path.join(IMAGES_DIR, "study_hours_vs_marks.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"[*] Saved Matplotlib regression chart to: {chart1_path}")

    # 2. Residuals Distribution Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(9, 4.5), dpi=150)
    plt.axhline(0, color="#64748B", linestyle="--", linewidth=1.2)
    plt.scatter(
        y_pred, 
        residuals, 
        color="#8B5CF6", 
        alpha=0.8, 
        edgecolors="#4C1D95", 
        s=65,
        label="Test Set Residuals"
    )
    plt.title("Residuals Analysis (Prediction Errors)", fontsize=13, fontweight="bold", color="#172B4D", pad=12)
    plt.xlabel("Predicted Marks", fontsize=11, fontweight="600", color="#172B4D")
    plt.ylabel("Residuals (Actual - Predicted)", fontsize=11, fontweight="600", color="#172B4D")
    plt.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0")
    plt.tight_layout()
    chart2_path = os.path.join(IMAGES_DIR, "residuals_plot.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"[*] Saved Matplotlib residuals chart to: {chart2_path}")


def train_and_evaluate():
    """Executes the training pipeline, saves model, metadata, and plots."""
    df = load_and_validate_dataset(DATASET_PATH)

    # Feature & Target Separation
    X = df[["Study_Hours"]]
    y = df["Marks"]

    # Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"[*] Train set size: {len(X_train)} samples")
    print(f"[*] Test set size: {len(X_test)} samples")

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model Evaluation on Test Set
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    print("=" * 60)
    print("MODEL EVALUATION RESULTS (Test Set):")
    print(f"  - Slope (m): {slope:.4f}")
    print(f"  - Intercept (c): {intercept:.4f}")
    print(f"  - Equation: Marks = {slope:.2f} * Study_Hours + {intercept:.2f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - R-squared (R2) Score: {r2:.4f} ({r2 * 100:.2f}% variance explained)")
    print("=" * 60)

    # Save trained model with Joblib
    joblib.dump(model, MODEL_PATH)
    print(f"[*] Trained model saved to: {MODEL_PATH}")

    # Generate Matplotlib plots
    generate_visualizations(df, model, X_test, y_test, y_pred)

    # Prepare detailed comparison for test set
    test_comparison = []
    for h, act, prd in zip(X_test["Study_Hours"], y_test, y_pred):
        test_comparison.append({
            "hours": round(float(h), 1),
            "actual": round(float(act), 1),
            "predicted": round(float(prd), 1),
            "error": round(float(prd - act), 1)
        })
    # Sort test comparison by hours
    test_comparison = sorted(test_comparison, key=lambda x: x["hours"])

    # Prepare scatter data for Chart.js
    scatter_data = [
        {"x": round(float(row["Study_Hours"]), 1), "y": round(float(row["Marks"]), 1)}
        for _, row in df.iterrows()
    ]

    # Calculate regression line endpoints
    x_min = float(df["Study_Hours"].min())
    x_max = float(df["Study_Hours"].max())
    y_min_line = round(slope * x_min + intercept, 2)
    y_max_line = round(slope * x_max + intercept, 2)

    # Assemble comprehensive metadata
    metadata = {
        "model_name": "Linear Regression",
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "equation": f"Marks = {slope:.2f} × Study_Hours + {intercept:.2f}",
        "metrics": {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "total_records": len(df)
        },
        "dataset_stats": {
            "total_records": len(df),
            "avg_hours": round(float(df["Study_Hours"].mean()), 2),
            "avg_marks": round(float(df["Marks"].mean()), 2),
            "min_hours": round(float(df["Study_Hours"].min()), 1),
            "max_hours": round(float(df["Study_Hours"].max()), 1),
            "min_marks": round(float(df["Marks"].min()), 1),
            "max_marks": round(float(df["Marks"].max()), 1),
            "std_hours": round(float(df["Study_Hours"].std()), 2),
            "std_marks": round(float(df["Marks"].std()), 2)
        },
        "regression_line": {
            "x_min": x_min,
            "y_at_x_min": y_min_line,
            "x_max": x_max,
            "y_at_x_max": y_max_line
        },
        "scatter_data": scatter_data,
        "test_comparison": test_comparison
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[*] Model metadata saved to: {METADATA_PATH}")

    return model, metadata


if __name__ == "__main__":
    train_and_evaluate()
