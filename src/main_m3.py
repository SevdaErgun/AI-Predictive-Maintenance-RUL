import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# --- CONFIGURATION ---
DATA_DIR = "data/raw" 
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FILE   = os.path.join(DATA_DIR, "RUL_FD001.txt")

# Remove constant sensors
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
WINDOW_SIZE = 30  # Sliding window size

def load_data():
    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    train = pd.read_csv(TRAIN_FILE, sep=r"\s+", header=None, names=cols)
    test  = pd.read_csv(TEST_FILE, sep=r"\s+", header=None, names=cols)
    rul   = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
    rul["unit"] = range(1, len(rul) + 1)
    return train, test, rul

def process_targets(train_df):
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    train_df = train_df.merge(max_cycle, on="unit")
    train_df["RUL"] = train_df["max_time"] - train_df["time"]
    train_df = train_df.drop(columns=["max_time"])
    return train_df

def add_rolling_features(df, sensor_cols, window_size):
    df_rolled = df.copy()
    
    # Group sensor data by unit and subject it to rolling processing.
    grouped = df_rolled.groupby("unit")
    
    for col in sensor_cols:
        # Average (Trend information)
        df_rolled[f"{col}_mean"] = grouped[col].transform(lambda x: x.rolling(window_size).mean())
        # Standard Deviation (Wave/Noise Information)
        df_rolled[f"{col}_std"] = grouped[col].transform(lambda x: x.rolling(window_size).std())
        
   
    # Select just rolling features and basic information. (Generally, only rolling features are used to reduce noise.)
    keep_cols = ["unit", "time", "os1", "os2", "os3", "RUL"] if "RUL" in df.columns else ["unit", "time", "os1", "os2", "os3"]
    feat_cols = [c for c in df_rolled.columns if c.endswith("_mean") or c.endswith("_std")]
    
    return df_rolled[keep_cols + feat_cols]

def prepare_train_data(train_df, sensor_cols):
    # Add rolling featurs
    train_proc = add_rolling_features(train_df, sensor_cols, WINDOW_SIZE)
    
    # Skip NaN rows (loops smaller than the window size)
    train_proc = train_proc.dropna()
    
    # Distinguishing between Feature and Target
    feature_cols = [c for c in train_proc.columns if c not in ["unit", "time", "RUL"]]
    X_train = train_proc[feature_cols]
    y_train = train_proc["RUL"]
    
    return X_train, y_train, feature_cols

def prepare_test_data(test_df, rul_df, sensor_cols, feature_cols):
    test_proc = add_rolling_features(test_df, sensor_cols, WINDOW_SIZE)
    
    # Just keep the last cycle for each unit
    # After the rolling process, the last line contains a summary of the past 30 cycles
    test_last = test_proc.groupby("unit").last().reset_index()
    
    # Sort by unit to align with RUL values
    test_last = test_last.sort_values("unit")
    rul_df = rul_df.sort_values("unit")
    
    
    X_test = test_last[feature_cols]
    y_test = rul_df["RUL"] # Real RUL values
    
    return X_test, y_test

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=300,       # Number of trees
        learning_rate=0.02,     # Learning more slowly and carefully.
        max_depth=6,            # Tree depth
        subsample=0.8,          # Data sampling to prevent overfitting.
        colsample_bytree=0.8,   # Feature sampling
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def explain_model(model, X_test):
    # take 100 random samples from the test set
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    print("\n--- SHAP Summary Plot Generating... ---")
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig("reports/images/shap_summary.png")
    print("Graphic saved: reports/images/shap_summary.png")

def plot_predictions(y_true, y_pred):
    print("\n--- Prediction vs. Real Value Graphic Generating... ---")
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.values, label="Real RUL", color='black', linestyle='-')
    plt.plot(y_pred, label="Prediction (XGBoost)", color='red', linestyle='--')
    plt.title("Test Set: Actual vs. Predicted RUL")
    plt.xlabel("Motor Index (Test Set)")
    plt.ylabel("RUL (Remaining Life)")
    plt.legend()
    plt.savefig("reports/images/pred_vs_actual.png")
    print("Graphic saved: reports/images/pred_vs_actual.png")

def plot_best_worst_cases(model, X_test, y_test):
    print("\n--- Best/Worst Case Graphic Generating... ---")

    # Get predictions
    preds = model.predict(X_test)
    
    # Calculate errors
    errors = np.abs(y_test - preds)
    
    # Find best and worst cases
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- BEST CASE ---
    axes[0].bar(["Real", "Predicted"], [y_test.iloc[best_idx], preds[best_idx]], color=['green', 'blue'])
    axes[0].set_title(f"Strength: Best Prediction (Motor ID: {y_test.index[best_idx]})\nError: {errors.iloc[best_idx]:.2f} cycle")
    axes[0].set_ylabel("RUL")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- WORST CASE ---
    axes[1].bar(["Real", "Predicted"], [y_test.iloc[worst_idx], preds[worst_idx]], color=['green', 'red'])
    axes[1].set_title(f"Weakness: Worst Prediction (Motor ID: {y_test.index[worst_idx]})\nError: {errors.iloc[worst_idx]:.2f} cycle")
    axes[1].set_ylabel("RUL")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("reports/images/best_worst_case.png")
    print("Graphic saved: reports/images/best_worst_case.png")
    
    return y_test.iloc[worst_idx], preds[worst_idx]
   
def main():
    train, test, rul = load_data()
    
    # Calculate RUL
    train = process_targets(train)
    
    sensor_cols = [c for c in train.columns if c.startswith("s") and c not in CONSTANT_SENSORS]
    
    print(f"Feature Engineering: Sliding Window (Size={WINDOW_SIZE})...")
    fe_start = time.time()
    X_train, y_train, feature_cols = prepare_train_data(train, sensor_cols)
    X_test, y_test = prepare_test_data(test, rul, sensor_cols, feature_cols)
    fe_time = time.time() - fe_start

    print(f"Model Training: XGBoost ({X_train.shape[0]} sample)...")
    train_start = time.time()

    model = train_xgboost(X_train, y_train)
    
    train_end = time.time()    
    training_duration = train_end - train_start

    sample_input = X_test.iloc[[0]] 
    
    inf_start = time.time()
    for _ in range(1000):
        model.predict(sample_input)
    inf_end = time.time()
    
    avg_inference_time = (inf_end - inf_start) / 1000

    # Değerlendirme
    print("\n--- Results for XGBoost + Windowing ---")
    preds_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds_test))
    mae = mean_absolute_error(y_test, preds_test)  
    r2 = r2_score(y_test, preds_test)
    
    print(f"Test RMSE : {rmse:.3f} (Goal: < 25.0)")
    print(f"Test MAE  : {mae:.3f}") 
    print(f"Test R²   : {r2:.3f}  (Goal: > 0.60)")
    
    print("\n--- Computation Durations ---")
    print(f"Model Training Duration : {training_duration:.2f} Second")
    print(f"Inference Duration: {avg_inference_time:.6f} Seconds (or {avg_inference_time*1000:.2f} ms)")
    print(f"Feature Engineering Duration: {fe_time:.2f} Second")

    # Explainability
    explain_model(model, X_test)
    
    plot_predictions(y_test, preds_test)

    plot_best_worst_cases(model, X_test, y_test)

if __name__ == "__main__":
    main()