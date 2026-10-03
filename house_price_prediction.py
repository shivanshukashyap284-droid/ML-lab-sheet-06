"""
Project 2: House Price Prediction System
Run in VS Code:
    python house_price_prediction.py

The program creates a sample housing dataset, compares regression
algorithms, evaluates them, and predicts the price of a new house.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

# 1. Create a sample housing dataset
n = 300
locations = np.random.choice(
    ["Urban", "Suburban", "Rural"], size=n, p=[0.45, 0.35, 0.20]
)

df = pd.DataFrame({
    "area_sqft": np.random.randint(600, 3501, n),
    "bedrooms": np.random.randint(1, 6, n),
    "bathrooms": np.random.randint(1, 5, n),
    "parking": np.random.randint(0, 4, n),
    "age_years": np.random.randint(0, 31, n),
    "location": locations
})

location_effect = df["location"].map({
    "Urban": 900000,
    "Suburban": 500000,
    "Rural": 150000
})

noise = np.random.normal(0, 150000, n)

df["price"] = (
    3500 * df["area_sqft"]
    + 300000 * df["bedrooms"]
    + 250000 * df["bathrooms"]
    + 120000 * df["parking"]
    - 50000 * df["age_years"]
    + location_effect
    + noise
).clip(500000, None).round(0)

print("\n=== House Price Prediction System ===")
print("\nDataset preview:")
print(df.head())

# 2. Features and target
X = df.drop(columns=["price"])
y = df["price"]

categorical_features = ["location"]
numeric_features = [
    "area_sqft", "bedrooms", "bathrooms", "parking", "age_years"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("location", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features)
    ]
)

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Compare regression algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, random_state=42
    )
}

results = {}

print("\n=== Model Evaluation ===")
for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    results[name] = (pipeline, r2)

    print(f"\n{name}")
    print(f"MAE  : Rs. {mae:,.2f}")
    print(f"RMSE : Rs. {rmse:,.2f}")
    print(f"R2   : {r2:.3f}")

# 5. Select the model with highest R2
best_name = max(results, key=lambda name: results[name][1])
best_model = results[best_name][0]
print(f"\nSelected model: {best_name}")

# 6. Predict price for a new house
new_house = pd.DataFrame([{
    "area_sqft": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "parking": 1,
    "age_years": 5,
    "location": "Urban"
}])

predicted_price = float(max(0, best_model.predict(new_house)[0]))

print("\n=== New House Price Prediction ===")
print(new_house.to_string(index=False))
print(f"\nPredicted House Price: Rs. {predicted_price:,.2f}")
print("\nProject completed successfully.")
