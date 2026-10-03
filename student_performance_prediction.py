"""
Project 1: Student Performance Prediction System
Run in VS Code:
    python student_performance_prediction.py

The program creates a sample dataset, trains multiple ML models,
evaluates them, and predicts the performance of a new student.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# 1. Create a sample dataset
n = 200
df = pd.DataFrame({
    "attendance": np.random.randint(55, 101, n),
    "internal_marks": np.random.randint(35, 96, n),
    "assignment_score": np.random.randint(35, 101, n),
    "study_hours": np.round(np.random.uniform(1, 8, n), 1),
    "previous_semester_result": np.random.uniform(40, 95, n).round(1)
})

# Target: final academic percentage
noise = np.random.normal(0, 3, n)
df["final_percentage"] = (
    0.18 * df["attendance"]
    + 0.32 * df["internal_marks"]
    + 0.16 * df["assignment_score"]
    + 2.0 * df["study_hours"]
    + 0.28 * df["previous_semester_result"]
    + noise
).clip(0, 100).round(2)

print("\n=== Student Performance Prediction System ===")
print("\nDataset preview:")
print(df.head())

# 2. Features and target
X = df.drop(columns=["final_percentage"])
y = df["final_percentage"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Train two regression algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    )
}

results = {}

print("\n=== Model Evaluation ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = (model, r2)
    print(f"\n{name}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.3f}")

# 5. Select model with highest R2
best_name = max(results, key=lambda name: results[name][1])
best_model = results[best_name][0]
print(f"\nSelected model: {best_name}")

# 6. Predict a new student's performance
new_student = pd.DataFrame([{
    "attendance": 88,
    "internal_marks": 78,
    "assignment_score": 85,
    "study_hours": 5.5,
    "previous_semester_result": 76
}])

prediction = best_model.predict(new_student)[0]
prediction = float(np.clip(prediction, 0, 100))

if prediction >= 75:
    category = "Excellent"
elif prediction >= 60:
    category = "Good"
elif prediction >= 50:
    category = "Average"
else:
    category = "Needs Improvement"

print("\n=== New Student Prediction ===")
print(new_student.to_string(index=False))
print(f"\nPredicted Final Percentage: {prediction:.2f}%")
print(f"Performance Category: {category}")
print("\nProject completed successfully.")
