"""
================================================================================
 DarkGuard ML Engine — Production-Grade Dark Pattern Detection System
================================================================================
 Architecture:
   - Ensemble Voting Classifier: Random Forest + Gradient Boosting + Logistic Regression
   - Cross-validated with Stratified K-Fold (k=5)
   - Automated Hyperparameter Tuning via RandomizedSearchCV
   - SMOTE oversampling for class balance
   - Full pipeline with preprocessing, feature engineering, and calibration
   - Model versioning and artifact saving
   - Detailed performance reporting (ROC-AUC, F1, Precision, Recall)
   - Autoscaling-ready: stateless predict() interface + joblib serialization
================================================================================
"""

import os
import sys
import time
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

from datetime import datetime
from pathlib import Path

# Scikit-learn core
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)

# Class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    warnings.warn("[WARN] imbalanced-learn not installed. SMOTE will be skipped.")

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION  (single source of truth — change here, nowhere else)
# -----------------------------------------------------------------------------

CONFIG = {
    "FEATURES": [
        "Fake_Urgency",
        "Scarcity",
        "Confusing_Text",
        "Hidden_Cost",
        "Forced_Action",
        "Social_Proof_Fake",
        "Misdirection",
        "Visual_Trick",
        "Confirmshaming",
        "Sneak_Into_Basket",
        "Roach_Motel",
        "Privacy_Zuckering",
        "Trick_Questions"
    ],
    # Domain-expert weights (used as an auxiliary engineered feature)
    "WEIGHTS": {
        "Fake_Urgency":      0.12,
        "Hidden_Cost":       0.12,
        "Sneak_Into_Basket": 0.12,
        "Roach_Motel":       0.10,
        "Scarcity":          0.08,
        "Privacy_Zuckering": 0.08,
        "Confusing_Text":    0.08,
        "Trick_Questions":   0.08,
        "Forced_Action":     0.06,
        "Misdirection":      0.05,
        "Confirmshaming":    0.05,
        "Social_Proof_Fake": 0.04,
        "Visual_Trick":      0.02,
    },
    "THRESHOLD":        0.50,       # Weighted score fallback threshold
    "RANDOM_STATE":     42,
    "CV_FOLDS":         5,          # Stratified K-Fold splits
    "TUNING_ITERS":     40,         # RandomizedSearchCV iterations
    "MIN_CONFIDENCE":   0.60,       # Below this → flag as "uncertain"
    "MODEL_VERSION":    "2.1.0",
    "OUTPUT_DIR":       "model_artifacts",
    "DATASET_FILES": {
        "train":     "training_dataset.xlsx",
        "test":      "testing_dataset.xlsx",
        "synthetic": "synthetic_dataset.xlsx",
    },
}

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("DarkGuard")


# -----------------------------------------------------------------------------
# STEP 1 — DATA LOADING & VALIDATION
# -----------------------------------------------------------------------------

class DataLoader:
    """Loads, validates, and merges datasets for training."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.features = cfg["FEATURES"]
        self.required_cols = self.features + ["Label"]

    def _load_excel(self, path: str, name: str) -> pd.DataFrame:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_excel(path, engine="openpyxl")
        log.info(f"Loaded {name}: {len(df):,} rows, {len(df.columns)} columns")
        return df

    def _validate(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        missing = [c for c in self.required_cols if c in df.columns is False]
        # Drop cols we don't need; keep only required
        available_features = [f for f in self.features if f in df.columns]

        if "Label" not in df.columns:
            # Auto-derive label from weighted score
            log.warning(f"{name}: 'Label' column missing — deriving from weighted score")
            w = np.array([self.cfg["WEIGHTS"][f] for f in available_features])
            df["Label"] = (df[available_features].values @ w >= self.cfg["THRESHOLD"]).astype(int)

        # Fill missing feature cols with 0
        for f in self.features:
            if f not in df.columns:
                log.warning(f"{name}: Feature '{f}' missing — filling with 0")
                df[f] = 0

        # Clip to binary 0/1
        df[self.features] = df[self.features].clip(0, 1).fillna(0).astype(int)
        df["Label"] = df["Label"].astype(int)
        return df

    def load_all(self):
        dfs = {}
        for key, fname in self.cfg["DATASET_FILES"].items():
            df = self._load_excel(fname, key)
            df = self._validate(df, key)
            dfs[key] = df

        # Merge train + synthetic into one training set
        train_full = pd.concat(
            [dfs["train"], dfs["synthetic"]], ignore_index=True
        ).drop_duplicates(subset=self.features)

        test_df = dfs["test"]

        log.info(
            f"Merged training set: {len(train_full):,} rows "
            f"(dark={sum(train_full['Label']==1)}, clean={sum(train_full['Label']==0)})"
        )
        log.info(
            f"Test set: {len(test_df):,} rows "
            f"(dark={sum(test_df['Label']==1)}, clean={sum(test_df['Label']==0)})"
        )
        return train_full, test_df


# -----------------------------------------------------------------------------
# STEP 2 — FEATURE ENGINEERING
# -----------------------------------------------------------------------------

class FeatureEngineer:
    """
    Adds derived features on top of the 8 raw binary signals.
    All features remain interpretable (no opaque transformations).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.features = cfg["FEATURES"]
        self.weights = cfg["WEIGHTS"]
        self.weight_arr = np.array([cfg["WEIGHTS"][f] for f in self.features])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X_raw = df[self.features].values.astype(float)

        # Feature 1-8: Raw binary signals
        # Feature 9: Weighted score (domain-expert knowledge injected into model)
        weighted_score = X_raw @ self.weight_arr

        # Feature 10: Count of patterns present (cardinality)
        pattern_count = X_raw.sum(axis=1)

        # Feature 11: High-severity pattern flag (any of the top-3 weighted patterns detected)
        top3 = ["Fake_Urgency", "Hidden_Cost", "Scarcity"]
        top3_idx = [self.features.index(f) for f in top3]
        high_severity = (X_raw[:, top3_idx].sum(axis=1) >= 2).astype(float)

        # Feature 12: Interaction — urgency AND scarcity together (strongest dark combo)
        urgency_scarcity = (
            X_raw[:, self.features.index("Fake_Urgency")] *
            X_raw[:, self.features.index("Scarcity")]
        )

        # Feature 13: Interaction — hidden cost AND forced action
        hidden_forced = (
            X_raw[:, self.features.index("Hidden_Cost")] *
            X_raw[:, self.features.index("Forced_Action")]
        )

        # Feature 14: Confusing text AND misdirection (deceptive language combo)
        confuse_misdirect = (
            X_raw[:, self.features.index("Confusing_Text")] *
            X_raw[:, self.features.index("Misdirection")]
        )

        # Feature 15: Roach Motel and Trick Questions 
        roach_trick = (
            X_raw[:, self.features.index("Roach_Motel")] *
            X_raw[:, self.features.index("Trick_Questions")]
        )

        # Feature 16: Sneak Into Basket and Hidden Cost
        sneak_hidden = (
            X_raw[:, self.features.index("Sneak_Into_Basket")] *
            X_raw[:, self.features.index("Hidden_Cost")]
        )

        # Stack all into final feature matrix
        X_engineered = np.column_stack([
            X_raw,
            weighted_score,
            pattern_count,
            high_severity,
            urgency_scarcity,
            hidden_forced,
            confuse_misdirect,
            roach_trick,
            sneak_hidden,
        ])

        return X_engineered

    @property
    def feature_names(self):
        return self.features + [
            "Weighted_Score",
            "Pattern_Count",
            "High_Severity_Flag",
            "Urgency_x_Scarcity",
            "Hidden_x_Forced",
            "Confuse_x_Misdirect",
            "Roach_x_Trick",
            "Sneak_x_Hidden",
        ]


# -----------------------------------------------------------------------------
# STEP 3 — MODEL DEFINITIONS & HYPERPARAMETER SEARCH SPACES
# -----------------------------------------------------------------------------

def build_base_models(cfg: dict):
    rs = cfg["RANDOM_STATE"]

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        random_state=rs,
        n_jobs=-1,
    )

    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        max_features="sqrt",
        min_samples_split=4,
        min_samples_leaf=2,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=rs,
    )

    lr = LogisticRegression(
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=rs,
    )

    return rf, gb, lr


RF_PARAM_SPACE = {
    "n_estimators":       [100, 150, 200],
    "max_depth":          [3, 4, 5, 6],
    "min_samples_split":  [4, 6, 8],
    "min_samples_leaf":   [2, 3, 4],
    "max_features":       ["sqrt", "log2", 0.5],
    "class_weight":       ["balanced", "balanced_subsample"],
}

GB_PARAM_SPACE = {
    "n_estimators":   [100, 150, 200],
    "max_depth":      [2, 3, 4],
    "learning_rate":  [0.01, 0.03, 0.05],
    "subsample":      [0.7, 0.8, 0.9],
    "max_features":   ["sqrt", "log2"],
}


# -----------------------------------------------------------------------------
# STEP 4 — HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------

def tune_model(model, param_space, X, y, cfg):
    log.info(f"  Tuning {model.__class__.__name__} ({cfg['TUNING_ITERS']} iterations)...")
    cv = StratifiedKFold(n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=cfg["RANDOM_STATE"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=cfg["TUNING_ITERS"],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=0,
        random_state=cfg["RANDOM_STATE"],
        refit=True,
    )
    search.fit(X_scaled, y)

    log.info(
        f"  Best params: {search.best_params_} | "
        f"Best CV F1: {search.best_score_:.4f}"
    )
    return search.best_estimator_


# -----------------------------------------------------------------------------
# STEP 5 — ENSEMBLE VOTING CLASSIFIER
# -----------------------------------------------------------------------------

def build_ensemble(rf_best, gb_best, lr_model, X_train, y_train, cfg):
    """
    Soft-voting ensemble: each estimator contributes probability estimates.
    Calibrated via Platt scaling so probabilities are well-calibrated.
    """
    log.info("Building Ensemble Voting Classifier (soft vote)...")

    # Wrap in Pipeline with scaler
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("clf", rf_best)])
    gb_pipe = Pipeline([("scaler", StandardScaler()), ("clf", gb_best)])
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", lr_model)])

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest",        rf_pipe),
            ("gradient_boosting",    gb_pipe),
            ("logistic_regression",  lr_pipe),
        ],
        voting="soft",
        weights=[3, 2, 1],      # RF gets highest weight — best on this data type
        n_jobs=-1,
        verbose=False,
    )

    # Calibrate to get reliable probabilities
    calibrated = CalibratedClassifierCV(ensemble, cv=3, method="sigmoid")
    calibrated.fit(X_train, y_train)

    log.info("Ensemble trained and calibrated.")
    return calibrated


# -----------------------------------------------------------------------------
# STEP 6 — CROSS-VALIDATION EVALUATION
# -----------------------------------------------------------------------------

def cross_validate_model(model, X, y, cfg):
    """Stratified 5-fold cross-validation across all key metrics."""
    cv = StratifiedKFold(n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=cfg["RANDOM_STATE"])

    scoring = {
        "accuracy":  "accuracy",
        "f1":        "f1",
        "precision": "precision",
        "recall":    "recall",
        "roc_auc":   "roc_auc",
    }

    log.info(f"Running {cfg['CV_FOLDS']}-fold cross-validation...")
    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )
    return results


def print_cv_results(results):
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  CROSS-VALIDATION RESULTS ({len(results['test_accuracy'])} folds)")
    print(sep)
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    print(f"  {'Metric':<18} {'Train Mean':>12} {'Test Mean':>12} {'Test Std':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")
    for m in metrics:
        train_mean = results[f"train_{m}"].mean()
        test_mean  = results[f"test_{m}"].mean()
        test_std   = results[f"test_{m}"].std()
        print(f"  {m.capitalize():<18} {train_mean:>12.4f} {test_mean:>12.4f} {test_std:>10.4f}")
    print(sep)


# -----------------------------------------------------------------------------
# STEP 7 — HOLDOUT TEST EVALUATION
# -----------------------------------------------------------------------------

def evaluate_on_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":          round(accuracy_score(y_test, y_pred),          4),
        "f1":                round(f1_score(y_test, y_pred),                 4),
        "precision":         round(precision_score(y_test, y_pred),          4),
        "recall":            round(recall_score(y_test, y_pred),             4),
        "roc_auc":           round(roc_auc_score(y_test, y_proba),           4),
        "avg_precision":     round(average_precision_score(y_test, y_proba), 4),
    }

    sep = "=" * 65
    print(f"\n{sep}")
    print("  HOLDOUT TEST PERFORMANCE")
    print(sep)
    print(f"  Accuracy          : {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score          : {metrics['f1']*100:.2f}%")
    print(f"  Precision         : {metrics['precision']*100:.2f}%")
    print(f"  Recall            : {metrics['recall']*100:.2f}%")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision     : {metrics['avg_precision']:.4f}")
    print(sep)

    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Dark Pattern"], digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"  +-------------------+-------------------+")
    print(f"  | True Neg :  {cm[0][0]:5d} | False Pos:  {cm[0][1]:5d} |")
    print(f"  +-------------------+-------------------+")
    print(f"  | False Neg:  {cm[1][0]:5d} | True Pos :  {cm[1][1]:5d} |")
    print(f"  +-------------------+-------------------+")

    return metrics


# -----------------------------------------------------------------------------
# STEP 8 — MODEL ARTIFACT SAVING
# -----------------------------------------------------------------------------

def save_artifacts(model, engineer, cfg, metrics):
    output_dir = Path(cfg["OUTPUT_DIR"])
    output_dir.mkdir(exist_ok=True)

    version    = cfg["MODEL_VERSION"]
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = output_dir / f"darkguard_model_v{version}.joblib"
    joblib.dump(model, model_path, compress=3)
    log.info(f"Model saved: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # Save feature engineer (needed for inference)
    fe_path = output_dir / f"feature_engineer_v{version}.joblib"
    joblib.dump(engineer, fe_path)
    log.info(f"Feature engineer saved: {fe_path}")

    # Save metadata JSON (for deployment / model registry)
    meta = {
        "model_version":    version,
        "trained_at":       timestamp,
        "features":         cfg["FEATURES"],
        "engineered_features": engineer.feature_names,
        "threshold":        cfg["THRESHOLD"],
        "min_confidence":   cfg["MIN_CONFIDENCE"],
        "test_metrics":     metrics,
        "model_path":       str(model_path),
        "feature_engineer_path": str(fe_path),
    }
    meta_path = output_dir / f"model_metadata_v{version}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved: {meta_path}")

    # Latest symlink (Windows-safe copy)
    latest_path = output_dir / "darkguard_model_LATEST.joblib"
    joblib.dump(model, latest_path, compress=3)

    latest_meta = output_dir / "model_metadata_LATEST.json"
    with open(latest_meta, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Latest model artifacts updated.")
    return str(model_path)


# -----------------------------------------------------------------------------
# STEP 9 — INFERENCE ENGINE (Production-Ready)
# -----------------------------------------------------------------------------

class DarkPatternPredictor:
    """
    Stateless, thread-safe inference class.
    Designed for deployment in autoscaling environments.
    Load once at startup, call predict() concurrently from multiple threads.

    Usage:
        predictor = DarkPatternPredictor.load("model_artifacts")
        result = predictor.predict({
            "Fake_Urgency": 1,
            "Hidden_Cost":  1,
            "Scarcity":     0,
            ...
        })
    """

    def __init__(self, model, engineer: FeatureEngineer, cfg: dict):
        self._model    = model
        self._engineer = engineer
        self._cfg      = cfg
        # Support both uppercase CONFIG keys and lowercase JSON metadata keys
        self._features        = cfg.get("FEATURES") or cfg.get("features") or CONFIG["FEATURES"]
        self._threshold_score = cfg.get("THRESHOLD") or cfg.get("threshold") or CONFIG["THRESHOLD"]
        self._min_confidence  = cfg.get("MIN_CONFIDENCE") or cfg.get("min_confidence") or CONFIG["MIN_CONFIDENCE"]
        # Weights: prefer CONFIG dict (richer); fall back to global CONFIG
        self._weights         = CONFIG["WEIGHTS"]

    @classmethod
    def load(cls, artifact_dir: str = "model_artifacts"):
        """
        Load from disk — call this once at server startup.
        FeatureEngineer is reconstructed from CONFIG (avoids pickle class-path issues).
        """
        d = Path(artifact_dir)
        model = joblib.load(d / "darkguard_model_LATEST.joblib")

        # Reconstruct engineer from current CONFIG — no pickle class dependency
        engineer = FeatureEngineer(CONFIG)

        with open(d / "model_metadata_LATEST.json") as f:
            meta = json.load(f)

        log.info(f"Model loaded: v{meta['model_version']} trained at {meta['trained_at']}")
        return cls(model, engineer, meta)

    def _validate_input(self, signals: dict) -> pd.DataFrame:
        row = {}
        for feat in self._features:
            val = signals.get(feat, 0)
            if val not in (0, 1):
                raise ValueError(f"Feature '{feat}' must be 0 or 1, got {val}")
            row[feat] = int(val)
        return pd.DataFrame([row])

    def predict(self, signals: dict) -> dict:
        """
        Args:
            signals: dict mapping each of the 8 features to 0 or 1.

        Returns:
            {
              "is_dark_pattern":  bool,
              "confidence":       float (0-1),
              "threat_level":     str  "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
              "weighted_score":   float,
              "pattern_count":    int,
              "detected_patterns": list[str],
              "certainty":        str  "CERTAIN" | "UNCERTAIN",
              "version":          str,
            }
        """
        df = self._validate_input(signals)
        X  = self._engineer.transform(df)

        prob         = float(self._model.predict_proba(X)[0][1])
        prediction   = int(self._model.predict(X)[0])
        weighted_score = float(
            (df[self._features].values @ np.array([self._weights[f] for f in self._features])).item()
        )

        detected_patterns = [f for f in self._features if signals.get(f, 0) == 1]
        pattern_count = len(detected_patterns)

        # User Request: If at least one pattern is found, force dark pattern = true
        if pattern_count >= 1:
            prediction = 1

        # Threat level classification
        if not prediction:
            threat_level = "NONE"
        elif prob < 0.50:
            threat_level = "LOW"
        elif prob < 0.70:
            threat_level = "MEDIUM"
        elif prob < 0.85:
            threat_level = "HIGH"
        else:
            threat_level = "CRITICAL"

        certainty = "CERTAIN" if abs(prob - 0.5) >= (0.5 - self._min_confidence) else "UNCERTAIN"

        return {
            "is_dark_pattern":   bool(prediction),
            "confidence":        round(prob, 4),
            "threat_level":      threat_level,
            "weighted_score":    round(weighted_score, 4),
            "pattern_count":     pattern_count,
            "detected_patterns": detected_patterns,
            "certainty":         certainty,
            "version":           self._cfg.get("model_version", "unknown"),
        }

    def predict_batch(self, records: list) -> list:
        """Batch prediction — efficient for bulk requests."""
        return [self.predict(r) for r in records]


# -----------------------------------------------------------------------------
# STEP 10 — FULL TRAINING PIPELINE
# -----------------------------------------------------------------------------

def run_training_pipeline():
    sep = "=" * 65
    print(f"\n{sep}")
    print("  DARKGUARD ML ENGINE — PRODUCTION TRAINING PIPELINE")
    print(f"  Version: {CONFIG['MODEL_VERSION']}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    t_start = time.time()

    # 1. Load data
    log.info("PHASE 1: Loading & validating datasets...")
    loader = DataLoader(CONFIG)
    train_df, test_df = loader.load_all()

    # 2. Feature engineering
    log.info("PHASE 2: Engineering features...")
    engineer = FeatureEngineer(CONFIG)
    X_train_raw = engineer.transform(train_df)
    X_test_raw  = engineer.transform(test_df)
    y_train = train_df["Label"].values
    y_test  = test_df["Label"].values

    log.info(f"  Engineered feature matrix shape: {X_train_raw.shape}")
    log.info(f"  Feature names: {engineer.feature_names}")

    # 3. SMOTE (if available)
    if SMOTE_AVAILABLE:
        log.info("PHASE 3: Applying SMOTE to balance classes...")
        smote = SMOTE(random_state=CONFIG["RANDOM_STATE"], k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train_raw, y_train)
        log.info(f"  Post-SMOTE: {sum(y_train==1)} dark / {sum(y_train==0)} clean")
    else:
        log.info("PHASE 3: SMOTE skipped (imbalanced-learn not installed).")
        X_train = X_train_raw

    X_test = X_test_raw

    # 4. Hyperparameter tuning
    log.info("PHASE 4: Hyperparameter tuning (RandomizedSearchCV)...")
    rf_base, gb_base, lr_base = build_base_models(CONFIG)
    rf_best = tune_model(rf_base, RF_PARAM_SPACE, X_train, y_train, CONFIG)
    gb_best = tune_model(gb_base, GB_PARAM_SPACE, X_train, y_train, CONFIG)

    # 5. Build ensemble
    log.info("PHASE 5: Building & calibrating ensemble...")
    ensemble = build_ensemble(rf_best, gb_best, lr_base, X_train, y_train, CONFIG)

    # 6. Cross-validation
    log.info("PHASE 6: Cross-validation evaluation...")
    cv_results = cross_validate_model(ensemble, X_train, y_train, CONFIG)
    print_cv_results(cv_results)

    # 7. Holdout test evaluation
    log.info("PHASE 7: Holdout test set evaluation...")
    metrics = evaluate_on_test(ensemble, X_test, y_test)

    # 8. Save artifacts
    log.info("PHASE 8: Saving model artifacts...")
    model_path = save_artifacts(ensemble, engineer, CONFIG, metrics)

    t_elapsed = time.time() - t_start
    print(f"\n{sep}")
    print(f"  TRAINING COMPLETE in {t_elapsed:.1f}s")
    print(f"  Model path : {model_path}")
    print(f"  Accuracy   : {metrics['accuracy']*100:.2f}%  |  F1: {metrics['f1']*100:.2f}%  |  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(sep)

    # 9. Quick inference demo
    print(f"\n  LIVE INFERENCE DEMO")
    print("-" * 65)
    predictor = DarkPatternPredictor.load(CONFIG["OUTPUT_DIR"])

    demo_cases = [
        {
            "name": "Amazon-style Flash Sale",
            "signals": {"Fake_Urgency":1,"Scarcity":1,"Hidden_Cost":1,"Social_Proof_Fake":1,"Misdirection":0,"Confusing_Text":0,"Forced_Action":0,"Visual_Trick":1}
        },
        {
            "name": "Fully Clean E-commerce",
            "signals": {"Fake_Urgency":0,"Scarcity":0,"Hidden_Cost":0,"Social_Proof_Fake":0,"Misdirection":0,"Confusing_Text":0,"Forced_Action":0,"Visual_Trick":0}
        },
        {
            "name": "Subscription Trap",
            "signals": {"Fake_Urgency":1,"Scarcity":0,"Hidden_Cost":1,"Social_Proof_Fake":0,"Misdirection":1,"Confusing_Text":1,"Forced_Action":1,"Visual_Trick":0}
        },
        {
            "name": "Max Dark Pattern",
            "signals": {"Fake_Urgency":1,"Scarcity":1,"Hidden_Cost":1,"Social_Proof_Fake":1,"Misdirection":1,"Confusing_Text":1,"Forced_Action":1,"Visual_Trick":1}
        },
        {
            "name": "Single Timer Only",
            "signals": {"Fake_Urgency":1,"Scarcity":0,"Hidden_Cost":0,"Social_Proof_Fake":0,"Misdirection":0,"Confusing_Text":0,"Forced_Action":0,"Visual_Trick":0}
        },
    ]

    print(f"  {'Page':<30} {'Verdict':<22} {'Confidence':>10} {'Threat':>10}")
    print(f"  {'-'*30} {'-'*22} {'-'*10} {'-'*10}")
    for case in demo_cases:
        result = predictor.predict(case["signals"])
        verdict = "DARK PATTERN" if result["is_dark_pattern"] else "CLEAN"
        print(
            f"  {case['name']:<30} {verdict:<22} "
            f"{result['confidence']*100:>9.1f}% {result['threat_level']:>10}"
        )

    print(f"\n  Full JSON output for last case:\n")
    print(json.dumps(result, indent=4))
    print(f"\n{sep}\n")

    return predictor


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    predictor = run_training_pipeline()
