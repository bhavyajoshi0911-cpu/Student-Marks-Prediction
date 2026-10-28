# Student Marks Prediction System — AI/ML Web Application

A modern, responsive, end-to-end Machine Learning web application and portfolio dashboard designed to forecast student exam marks based on daily study hours using Linear Regression.

Built with **Python**, **Pandas**, **NumPy**, **Matplotlib**, **Scikit-learn**, **Flask**, **Joblib**, and **Chart.js**.

---

## 📸 Overview & Features

- **Executive Analytics Dashboard:**
  - 4 Key Metric Stat Cards: Total Records, Average Study Hours, Average Marks, and Test R² Score.
  - Interactive Chart.js scatter plot of actual student scores with the fitted Linear Regression line.
  - Dataset Preview featuring the first 5 records with performance badges.
  - Model Performance Evaluation summary card with MAE and R² explanations.
  - Visual 9-stage engineering workflow explaining the end-to-end ML lifecycle.
- **Dynamic Asynchronous Prediction Panel:**
  - Instant prediction without page reload using `fetch` API.
  - Robust client-side and backend validation (empty, negative, non-numeric, >24 hrs).
  - Clean animated progress meter, grade categorization (A+, A, B, C, D, Fail), and personalized academic feedback.
  - Sensible educational boundaries ($0 \le \text{Marks} \le 100$).
- **Data Overview Page (`/data-overview`):**
  - Full dataset view of all 100 student records.
  - Real-time client-side search and filtering by Student ID or study hours.
  - Descriptive statistics table (Count, Mean, Std, Min, 25%, Median, 75%, Max).
  - One-click CSV dataset download.
- **Model Performance Page (`/model-performance`):**
  - Unseen 20% test split evaluation: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and R² Score.
  - Static high-resolution Matplotlib charts: Regression Fit and Residuals Distribution.
  - Row-by-row Actual vs. Predicted comparison table with individual prediction error metrics.
- **About Project Page (`/about`):**
  - Breakdown of mathematical formulation ($y = mx + c$).
  - Detailed explanation of each library's role in the architecture.
  - Transparent discussion of dataset assumptions and model limitations.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core programming and computational logic |
| **Data Manipulation** | Pandas | CSV loading, null validation, schema standardization, descriptive stats |
| **Numerical Computing**| NumPy | Array transformations, boundary clipping, root mean squared computations |
| **Visualization** | Matplotlib & Seaborn | Static regression plots and residual error distribution plots |
| **Machine Learning** | Scikit-learn | `LinearRegression`, `train_test_split`, MAE, MSE, and R² evaluation |
| **Model Persistence** | Joblib | High-efficiency serialization (`model.pkl`) and loading |
| **Web Framework** | Flask | HTTP request routing, Jinja2 template rendering, RESTful prediction API |
| **Frontend UI** | HTML5, CSS3, JavaScript | SaaS-inspired responsive layout, CSS grid/flexbox, animations |
| **Interactive Charts** | Chart.js 4.4 | Client-side responsive interactive scatter plot with hover tooltips |

---

## 📂 Project Structure

```
Student-Marks-Prediction/
├── app.py                      # Flask web application & REST API routes
├── train_model.py              # ML training, data validation, metrics & Matplotlib generator
├── dataset.csv                 # 100 realistic student performance records
├── model.pkl                   # Serialized Linear Regression model (Joblib)
├── model_metadata.json         # Evaluation metrics, coefficients, and distribution stats
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & execution guide
├── templates/
│   ├── base.html               # Shared layout, dark navy sidebar, header & footer
│   ├── index.html              # Main dashboard (stats, chart, prediction, preview, workflow)
│   ├── data_overview.html      # Full dataset table, statistics summary, search & CSV download
│   ├── model_performance.html  # In-depth test metrics, residuals & actual vs. predicted table
│   ├── about.html              # Architecture, tech stack breakdown, formula & ethics
│   └── 404.html                # Custom 404 error page
└── static/
    ├── css/
    │   └── style.css           # Premium SaaS styles, color palette, responsive design
    ├── js/
    │   └── script.js           # AJAX prediction handler, Chart.js setup, table filter
    └── images/
        ├── study_hours_vs_marks.png  # Generated Matplotlib regression plot
        └── residuals_plot.png        # Generated Matplotlib residuals analysis plot
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
Ensure Python 3.10 or newer is installed on your system.

### 2. Clone or Navigate to the Project Folder
```bash
cd "c:\Users\DELL\OneDrive\Desktop\Python Projects\Student-Marks-Prediction"
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model & Generate Visualizations
Run the standalone training pipeline. This will inspect the dataset, split the train/test sets, fit the Linear Regression model, evaluate test performance, save `model.pkl`, write `model_metadata.json`, and output Matplotlib charts to `static/images/`:
```bash
python train_model.py
```

*Example Terminal Output:*
```text
[*] Loading dataset from: dataset.csv
[*] Raw Dataset Shape: 100 rows, 3 columns
[*] Cleaned Dataset Shape: 100 valid rows
[*] Train set size: 80 samples
[*] Test set size: 20 samples
============================================================
MODEL EVALUATION RESULTS (Test Set):
  - Slope (m): 7.7486
  - Intercept (c): 16.1893
  - Equation: Marks = 7.75 * Study_Hours + 16.19
  - Mean Absolute Error (MAE): 4.4970
  - Mean Squared Error (MSE): 29.4373
  - Root Mean Squared Error (RMSE): 5.4256
  - R-squared (R2) Score: 0.9370 (93.70% variance explained)
============================================================
[*] Trained model saved to: model.pkl
[*] Saved Matplotlib regression chart to: static/images/study_hours_vs_marks.png
[*] Saved Matplotlib residuals chart to: static/images/residuals_plot.png
[*] Model metadata saved to: model_metadata.json
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🧪 Testing a Prediction

1. On the main Dashboard, locate the **Predict Your Marks** card on the right.
2. In the **Study Hours** input, enter a number (e.g. `6.5`).
3. Click the **Predict Marks** button.
4. The button displays a loading spinner and instantaneously communicates with `/predict`.
5. The green result card appears smoothly:
   - **Predicted Exam Marks:** `66.6 / 100`
   - **Progress Meter:** Animates to 66.6% width.
   - **Academic Tier:** `B (Good)`
   - **Feedback:** `👍 Decent grasp of concepts. Consistent daily review will boost your score.`
6. Test Edge Cases & Validation:
   - Leave input blank and click predict ➔ Displays inline validation error: *"Please enter the number of study hours."*
   - Enter `-3` ➔ Displays error: *"Study hours cannot be negative."*
   - Enter `28` ➔ Displays error: *"Study hours cannot exceed 24 hours in a single day."*
   - Enter `12` ➔ Safely predicts high score capped at `100 / 100` with *"A+ (Outstanding)"* rating.

---

## 🧠 How the Machine Learning Model Works

1. **Ordinary Least Squares (OLS):**
   The model fits a straight line through the observed data points by minimizing the sum of squared differences (residuals) between observed exam marks ($y$) and predicted marks ($\hat{y}$):
   $$\hat{y} = m \cdot X + c$$
   - **Slope ($m \approx 7.75$):** For each additional hour studied per day, the student's exam score increases on average by ~7.75 marks.
   - **Intercept ($c \approx 16.19$):** The baseline expected marks for a student who recorded 0 daily study hours.
2. **Train/Test Validation:**
   - 80% of samples (80 records) were used for model fitting.
   - 20% of samples (20 records) were strictly held out to test generalization on previously unseen data.
3. **Evaluation Metrics:**
   - **Mean Absolute Error (MAE = 4.50):** On average, the model's predictions differ from actual test marks by 4.5 marks.
   - **$R^2$ Score (0.937):** 93.7% of the total variance in student exam scores is explained by study hours alone.

---

## ⚠️ Limitations & Ethical Considerations

- **Univariate Simplicity:** Study hours alone do not determine academic success. Sleep hygiene, syllabus complexity, prior academic grounding, and testing anxiety play substantial roles.
- **Diminishing Marginal Utility:** In real life, studying 14+ hours consecutively creates cognitive fatigue and diminishing returns, which a linear model does not naturally model without polynomial or logarithmic terms.
- **Educational Intent:** This application is built as an educational demonstration of end-to-end Machine Learning deployment and should not be used as an absolute metric for student evaluation.
