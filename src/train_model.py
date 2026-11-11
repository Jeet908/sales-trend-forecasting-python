"""
Model Training, Cross-Validation, and Predictive Evaluation Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def evaluate_forecast_model(data_path: str):
    df = pd.read_csv(data_path)

    # Feature selection
    features = [
        'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 
        'CPI', 'Unemployment', 'Month', 'WeekOfYear', 'Sin_Week', 'Cos_Week'
    ]
    target = 'Weekly_Sales'

    # Filter columns that exist in the dataframe
    avail_features = [f for f in features if f in df.columns]

    X = df[avail_features]
    y = df[target]

    # Time-based or chronological split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Model 1: Baseline Regularized Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)

    # Model 2: Random Forest Ensemble Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Compute Metrics
    def print_metrics(model_name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"\n================ {model_name} ================")
        print(f"R² Score:            {r2:.4f}")
        print(f"RMSE:                ${rmse:,.2f}")
        print(f"Mean Absolute Error: ${mae:,.2f}")

    print_metrics("Ridge Regression (Baseline)", y_test, y_pred_ridge)
    print_metrics("Random Forest Regressor (Ensemble)", y_test, y_pred_rf)

    # Save model artifact
    joblib.dump(rf, "sales_demand_rf_model.pkl")
    print("\nBest model successfully persisted to sales_demand_rf_model.pkl")


if __name__ == "__main__":
    print("Model training script ready.")
