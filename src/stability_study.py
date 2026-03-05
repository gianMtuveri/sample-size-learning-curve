import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Train/test split (avoid evaluating on the same data you trained on)
from sklearn.model_selection import train_test_split

# ColumnTransformer + Pipeline let you:
# - apply different preprocessing to numeric vs categorical columns
# - keep everything reproducible and refit correctly inside bootstrap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# OneHotEncoder converts strings like "Male" / "flat" / "asymptomatic" to numeric columns
# StandardScaler helps logistic regression when numeric features have different scales
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

# SimpleImputer fills missing values so the model doesn’t crash
from sklearn.impute import SimpleImputer

# Logistic regression model
from sklearn.linear_model import LogisticRegression

# Bootstrap sampling with replacement
from sklearn.utils import resample

# Metrics: accuracy is OK but ROC AUC is usually better for medical risk
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, brier_score_loss


DATA_PATH = "data/heart_disease_uci.csv"


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV into a DataFrame.

    Note:
    - If your file contains '?' as missing values (some UCI variants do),
      use: pd.read_csv(path, na_values=["?"])
    """
    return pd.read_csv(path,
        true_values=["TRUE", "True", "true"],
        false_values=["FALSE", "False", "false"])




def add_target_and_drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Create binary target from num > 0
    2) Drop non-feature columns (num itself, id, dataset if present/constant)
    3) Convert TRUE/FALSE strings to 1/0 for boolean columns
    """
    df = df.copy()  # never mutate original df in place in a pipeline script

    # ---- target engineering ----
    # "num" is the UCI severity label; we binarize it:
    df["target"] = (df["num"] > 0).astype(int)

    # ---- drop columns we shouldn't model ----
    drop_cols = ["num"]  # remove original multi-class target
    if "id" in df.columns:
        drop_cols.append("id")  # identifier, not a feature
    if "dataset" in df.columns:
        # in your sample it's always "Cleveland", so it carries no information
        drop_cols.append("dataset")

    df = df.drop(columns=drop_cols)

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build a preprocessing + model pipeline:
    - numeric: impute median + scale
    - categorical: impute most frequent + one-hot encode
    - logistic regression on top

    This is the key fix: it turns your mixed-type DataFrame into a numeric matrix.
    """
    # Identify boolean vs numeric vs categorical columns from the DataFrame dtype
    bool_cols = X.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Numeric preprocessing:
    # - fill missing values with median
    # - standardize features (mean 0, std 1)
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Boolean preprocessing:
    # - impute with most frequent (median makes no sense for all-missing bools)
    # - convert to int (True/False -> 1/0) after imputation
    bool_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_int", FunctionTransformer(lambda a: a.astype(int))),
    ])
    


    # Categorical preprocessing:
    # - fill missing values with most frequent category
    # - one-hot encode (creates 0/1 columns for each category)
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ColumnTransformer applies different preprocessors to different column sets
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("bool", bool_pipe, bool_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )

    # Logistic regression:
    # - class_weight balanced helps if target classes are not 50/50
    # - max_iter increased to help convergence after one-hot
    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    # Full pipeline
    return Pipeline(steps=[
        ("preprocess", pre),
        ("model", model),
    ])


def stratified_subsample(X_train, y_train, n, rng):
    # compute per-class counts proportional to training prevalence
    classes, counts = np.unique(y_train, return_counts=True)
    proportions = counts / counts.sum()

    # initial allocation
    n_per_class = np.floor(proportions * n).astype(int)

    # fix rounding to ensure sum exactly n
    while n_per_class.sum() < n:
        # add remaining samples to the class with largest fractional remainder
        remainders = proportions * n - np.floor(proportions * n)
        n_per_class[np.argmax(remainders)] += 1

    idxs = []
    for cls, k in zip(classes, n_per_class):
        cls_idx = y_train[y_train == cls].index
        chosen = rng.choice(cls_idx, size=k, replace=False)
        idxs.append(chosen)

    idx = np.concatenate(idxs)

    rng.shuffle(idx)
    return X_train.loc[idx], y_train.loc[idx]


def run_experiment(X_train, X_test, y_train, y_test, n, rng):  
    # Fit + evaluate on test split

    X_sub, y_sub = stratified_subsample(X_train, y_train, n, rng)

    pipe = build_pipeline(X_sub)
    pipe.fit(X_sub, y_sub)
    
    proba_test = pipe.predict_proba(X_test)[:, 1]

    # Compute AUC and Brier score
    auc = roc_auc_score(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)
    
    auc_mean, auc_low, auc_high, brier_mean, brier_low, brier_high = bootstrap_metrics(pipe, X_train, y_train, X_test, y_test, 100, rng)
    
    print(f"n={n} | AUC mean={auc_mean:.4f} | AUC low={auc_low:.4f} | AUC high={auc_high:.4f} | Brier mean={brier_mean:.4f} | Brier low={brier_low:.4f} | Brier high={brier_high:.4f}")
    return auc_mean, auc_low, auc_high, brier_mean, brier_low, brier_high


def bootstrap_metrics(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iterations,
    rng
) -> np.ndarray:
    """
    Bootstrap AUC:
    - resample TRAIN set with replacement
    - refit pipeline on each bootstrap sample (important!)
    - evaluate AUC on the SAME fixed test set (honest comparison)
    - return distribution of AUCs so you can compute CI
    """
    aucs = []
    briers = []
    n = len(X_train)

    for _ in range(n_iterations):
        # Sample indices with replacement (size n)
        idx = rng.randint(0, n, size=n)

        # Subset rows; .iloc keeps alignment between X and y
        X_sample = X_train.iloc[idx]
        y_sample = y_train.iloc[idx]
    
        # Fit on bootstrap sample (refit preprocessors + model!)
        pipe.fit(X_sample, y_sample)

        proba = pipe.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
        briers.append(brier_score_loss(y_test, proba))

    aucs = np.array(aucs, dtype=float)
    briers = np.array(briers, dtype=float)

    auc_mean = float(aucs.mean())
    auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])

    brier_mean = float(briers.mean())
    brier_low, brier_high = np.percentile(briers, [2.5, 97.5])

    return auc_mean, float(auc_low), float(auc_high), brier_mean, float(brier_low), float(brier_high)


def main():
    rng = np.random.RandomState(42)
    
    # Load raw data
    df = load_data(DATA_PATH)
    print("Data loaded")

    # Create target, drop id/dataset/num, map TRUE/FALSE -> 1/0
    df = add_target_and_drop_cols(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    n_samp = [50, 100, 200, 400, len(X_train)]

    # List of aucs for different n
    rows = []
    
    for n in n_samp:
        auc_mean, auc_low, auc_high, brier_mean, brier_low, brier_high = run_experiment(X_train, X_test, y_train, y_test, n, rng)
        
        rows.append({
            "n_train": n,
            "auc_mean": auc_mean,
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
            "brier_mean": brier_mean,
            "brier_ci_low": brier_low,
            "brier_ci_high": brier_high,
        })

    summary_df = pd.DataFrame(rows)

    os.makedirs("results", exist_ok=True)
    out_csv = "results/sample_size_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print(summary_df)

    # AUC plot
    plt.figure()
    plt.plot(summary_df["n_train"], summary_df["auc_mean"])
    plt.fill_between(
        summary_df["n_train"],
        summary_df["auc_ci_low"],
        summary_df["auc_ci_high"],
        alpha=0.3
    )
    plt.xlabel("Training Sample Size")
    plt.ylabel("ROC AUC")
    plt.title("AUC vs Training Size")
    plt.savefig("results/fig_auc_vs_n.png")
    plt.close()
 
    # Brier plot
    plt.figure()
    plt.plot(summary_df["n_train"], summary_df["brier_mean"])
    plt.fill_between(
        summary_df["n_train"],
        summary_df["brier_ci_low"],
        summary_df["brier_ci_high"],
        alpha=0.3
    )
    plt.xlabel("Training Sample Size")
    plt.ylabel("Brier Score")
    plt.title("Brier Score vs Training Size")
    plt.savefig("results/fig_brier_vs_n.png")
    plt.close()

if __name__ == "__main__":
    main()
