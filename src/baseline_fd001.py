import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler



DATA_DIR = "data/raw" 

TRAIN_FILE = os.path.join(DATA_DIR, "train_FD001.txt")
TEST_FILE  = os.path.join(DATA_DIR, "test_FD001.txt")
RUL_FILE   = os.path.join(DATA_DIR, "RUL_FD001.txt")

CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]


def load_fd001_data():
    """Load FD001 dataset with correct 26-column structure."""

    colnames = ["unit", "time", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]

    train = pd.read_csv(TRAIN_FILE, sep=r"\s+", header=None, names=colnames)
    test  = pd.read_csv(TEST_FILE,  sep=r"\s+", header=None, names=colnames)

    rul = pd.read_csv(RUL_FILE, header=None, names=["RUL"])
    rul["unit"] = range(1, len(rul) + 1)

    return train, test, rul


def compute_rul_for_train(train_df):
    """Compute RUL for each training sample."""
    max_cycle = train_df.groupby("unit")["time"].max().reset_index()
    max_cycle.columns = ["unit", "max_time"]

    train_df = train_df.merge(max_cycle, on="unit")
    train_df["RUL"] = train_df["max_time"] - train_df["time"]
    train_df = train_df.drop(columns=["max_time"])

    return train_df


def prepare_features(train_df, test_df, rul_df):
    """Prepare matrices for training and testing."""

    train_df = train_df.drop(columns=CONSTANT_SENSORS)
    test_df  = test_df.drop(columns=CONSTANT_SENSORS)

    feature_cols = ["os1", "os2", "os3"] + [c for c in train_df.columns if c.startswith("s")]

    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values

    last_cycle = test_df.groupby("unit")["time"].max().reset_index()
    test_last = test_df.merge(last_cycle, on=["unit", "time"])

    # alignment (safe)
    test_last = test_last.sort_values("unit")
    rul_df = rul_df.sort_values("unit")

    X_test = test_last[feature_cols].values
    y_test = rul_df["RUL"].values

    return X_train, y_train, X_test, y_test, feature_cols


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


def train_baseline(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, label):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    print(f"\n--- Evaluation: {label} ---")
    print(f"RMSE : {rmse:.3f}")
    print(f"MAE  : {mae:.3f}")
    print(f"R²   : {r2:.3f}")

    return rmse, mae, r2


def main():
    print("Loading dataset...")
    train_df, test_df, rul_df = load_fd001_data()

    print("Computing RUL...")
    train_df = compute_rul_for_train(train_df)

    print("Preparing matrices...")
    X_train, y_train, X_test, y_test, _ = prepare_features(train_df, test_df, rul_df)

    print("Scaling...")
    X_train_s, X_test_s = scale_data(X_train, X_test)

    print("Training baseline model (RandomForest)...")
    model = train_baseline(X_train_s, y_train)

    evaluate(model, X_train_s, y_train, "Train (all cycles)")
    evaluate(model, X_test_s,  y_test,  "Test (last cycle RUL)")

    print("\nBaseline completed successfully.")


if __name__ == "__main__":
    main()
