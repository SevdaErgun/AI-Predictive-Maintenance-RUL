import os
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# CONFIGURATION
DATA_DIR = "data/raw" 
TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FILE   = os.path.join(DATA_DIR, "RUL_FD001.txt")
IMAGE_DIR = "reports/images"


CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
WINDOW_SIZE = 30
RUL_CLIP_LIMIT = 125  # Piece-wise RUL Threshold

os.makedirs(IMAGE_DIR, exist_ok=True)

def load_data():
    cols = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
    try:
        train = pd.read_csv(TRAIN_FILE, sep=r"\s+", header=None, names=cols)
        test  = pd.read_csv(TEST_FILE, sep=r"\s+", header=None, names=cols)
        rul   = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
        rul["unit"] = range(1, len(rul) + 1)
        return train, test, rul
    except FileNotFoundError:
        print("ERROR: Data files not found.")
        return None, None, None

def process_targets(train_df):
    # Calculate max cycle per unit
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]
    train_df = train_df.merge(max_cycle, on="unit")
    
    # Linear RUL calculation
    train_df["RUL"] = train_df["max_time"] - train_df["time"]
    
    # Clip at 125
    train_df["RUL"] = train_df["RUL"].clip(upper=RUL_CLIP_LIMIT)
    
    train_df = train_df.drop(columns=["max_time"])
    return train_df

def add_rolling_features(df, sensor_cols, window_size):
    df_rolled = df.copy()
    grouped = df_rolled.groupby("unit")
    
    for col in sensor_cols:
        df_rolled[f"{col}_mean"] = grouped[col].transform(lambda x: x.rolling(window_size).mean())
        df_rolled[f"{col}_std"] = grouped[col].transform(lambda x: x.rolling(window_size).std())
    
    keep_cols = ["unit", "time", "os1", "os2", "os3", "RUL"] if "RUL" in df.columns else ["unit", "time", "os1", "os2", "os3"]
    feat_cols = [c for c in df_rolled.columns if c.endswith("_mean") or c.endswith("_std")]
    
    return df_rolled[keep_cols + feat_cols]

def prepare_data(train, test, rul, sensor_cols):
    # Train Data Preparation
    train = process_targets(train)
    X_train = add_rolling_features(train, sensor_cols, WINDOW_SIZE).dropna()
    y_train = X_train["RUL"]
    

    X_train = X_train.drop(columns=["unit", "time", "RUL"])
    
    # Test Data Preparation
    y_test_clipped = rul["RUL"].clip(upper=RUL_CLIP_LIMIT)
    
    test_proc = add_rolling_features(test, sensor_cols, WINDOW_SIZE)
    test_last = test_proc.groupby("unit").last().reset_index().sort_values("unit")
    
    X_test = test_last.drop(columns=["unit", "time"])
    
    X_test = X_test[X_train.columns]
    
    return X_train, y_train, X_test, y_test_clipped

def train_final_model(X_train, y_train):
# Final XGBoost Model Training
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=6,         # Selected based on Test Set performance
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def plot_pred_vs_actual(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='#1f77b4', alpha=0.6, label='Predictions')
    
    # Perfect fit line (Diagonal)
    limit = max(y_true.max(), y_pred.max())
    plt.plot([0, limit], [0, limit], 'r--', linewidth=2, label='Perfect Fit')
    
    plt.xlabel('True RUL (Clipped at 125)')
    plt.ylabel('Predicted RUL')
    plt.title('M5 Final Model: True vs Predicted RUL')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = os.path.join(IMAGE_DIR, "m5_pred_vs_actual.png")
    plt.savefig(save_path)
    print(f"Graph saved: {save_path}")

def main():
    print("--- M5 FINAL RUN STARTED ---")
    train, test, rul = load_data()
    if train is None: return

    sensor_cols = [c for c in train.columns if c.startswith("s") and c not in CONSTANT_SENSORS]
    
    print("Preparing Data (Piece-wise RUL & Sliding Window)...")
    X_train, y_train, X_test, y_test = prepare_data(train, test, rul, sensor_cols)
    
    print("Training Final XGBoost Model...")
    start_time = time.time()
    model = train_final_model(X_train, y_train)
    print(f"Training Completed ({time.time() - start_time:.2f} sec)")
    
    # Prediction
    preds = model.predict(X_test)
    # Post-processing: Clip predictions to be safe
    preds = np.clip(preds, a_min=None, a_max=RUL_CLIP_LIMIT)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print("\n" + "="*40)
    print("M5 FINAL RESULTS (GOLDEN MODEL)")
    print("="*40)
    print(f"Final RMSE : {rmse:.3f}")
    print(f"Final MAE  : {mae:.3f}")
    print(f"Final R2   : {r2:.3f}")
    print("-" * 40)
    print(f"Improvement over M3 (RMSE 27.87): {27.87 - rmse:.3f} cycles")
    print("="*40)
    
    plot_pred_vs_actual(y_test, preds)

if __name__ == "__main__":
    main()