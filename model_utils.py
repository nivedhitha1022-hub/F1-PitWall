"""
model_utils.py
──────────────
Feature engineering, Random Forest churn classifier, and KMeans segmentation.
Built entirely against the real PitWall dataset column names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    subs: pd.DataFrame,
    sess: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregates session-level data per subscriber and merges with subscriber
    attributes to produce a flat feature matrix ready for ML.

    Parameters
    ----------
    subs : Subscribers DataFrame (800 rows)
    sess : Engagement Sessions DataFrame (29 240 rows)

    Returns
    -------
    df : merged feature matrix (800 rows, 30+ cols)
    """
    # ── Per-subscriber session aggregates ────────────────────────────────────
    agg = (
        sess
        .groupby("Subscriber Id")
        .agg(
            total_sessions       = ("Session Duration Min", "count"),
            avg_engagement       = ("Engagement Score",     "mean"),
            std_engagement       = ("Engagement Score",     "std"),
            avg_duration         = ("Session Duration Min", "mean"),
            total_duration       = ("Session Duration Min", "sum"),
            mobile_sessions      = ("Device",         lambda x: (x == "Mobile").sum()),
            weekend_sessions     = ("Is Weekend",      "sum"),
            high_eng_sessions    = ("Engagement Tier", lambda x: (x == "High").sum()),
            medium_eng_sessions  = ("Engagement Tier", lambda x: (x == "Medium").sum()),
        )
        .reset_index()
    )

    agg["mobile_pct"]   = (agg["mobile_sessions"]  / agg["total_sessions"]).round(4)
    agg["weekend_pct"]  = (agg["weekend_sessions"] / agg["total_sessions"]).round(4)
    agg["high_eng_pct"] = (agg["high_eng_sessions"]/ agg["total_sessions"]).round(4)
    agg["std_engagement"] = agg["std_engagement"].fillna(0)

    # Most-used content type
    top_content = (
        sess.groupby("Subscriber Id")["Content Type"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"Content Type": "top_content"})
    )

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = subs.merge(agg,         on="Subscriber Id", how="left")
    df = df.merge(top_content,   on="Subscriber Id", how="left")

    # Fill subscribers with zero logged sessions (shouldn't occur on this dataset)
    _fill_zero = ["total_sessions", "avg_engagement", "std_engagement",
                  "avg_duration", "total_duration", "mobile_pct",
                  "weekend_pct", "high_eng_pct"]
    for c in _fill_zero:
        df[c] = df[c].fillna(0)
    df["top_content"] = df["top_content"].fillna("Unknown")

    # ── Label-encode categoricals ──────────────────────────────────────────────
    cat_map = {
        "Plan":                "plan_enc",
        "Region":              "region_enc",
        "Acquisition Channel": "channel_enc",
        "Age Group":           "age_group_enc",
        "top_content":         "content_enc",
    }
    for src, dst in cat_map.items():
        df[dst] = LabelEncoder().fit_transform(df[src].astype(str))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CHURN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "plan_enc",
    "Monthly Price Usd",
    "region_enc",
    "channel_enc",
    "Age",
    "age_group_enc",
    "Tenure Months",
    "Renewal Count",
    "Nps Score",
    "total_sessions",
    "avg_engagement",
    "std_engagement",
    "avg_duration",
    "mobile_pct",
    "weekend_pct",
    "high_eng_pct",
    "content_enc",
]

FEATURE_LABELS = [
    "Plan Tier",
    "Monthly Price",
    "Region",
    "Acquisition Channel",
    "Age",
    "Age Group",
    "Tenure (months)",
    "Renewal Count",
    "NPS Score",
    "Total Sessions",
    "Avg Engagement Score",
    "Engagement Variability",
    "Avg Session Duration",
    "Mobile Usage %",
    "Weekend Usage %",
    "High-Engagement Session %",
    "Top Content Type",
]


def train_churn_model(df: pd.DataFrame):
    """
    Train a Random Forest churn classifier.

    Returns
    -------
    clf            : fitted RandomForestClassifier
    X_train, X_test, y_train, y_test
    y_pred         : predictions on test set
    y_prob         : probability of churn on test set
    importance_df  : DataFrame[feature, importance] sorted ascending
    df_scored      : full df with churn_prob & churn_pred appended
    """
    X = df[FEATURE_COLS].fillna(0)
    y = df["churn_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    importance_df = (
        pd.DataFrame({"feature": FEATURE_LABELS,
                      "importance": clf.feature_importances_})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    df_scored = df.copy()
    df_scored["churn_prob"] = clf.predict_proba(X)[:, 1]
    df_scored["churn_pred"] = clf.predict(X)

    return clf, X_train, X_test, y_train, y_test, y_pred, y_prob, importance_df, df_scored


def get_model_metrics(y_test, y_pred, y_prob) -> dict:
    """Compute all evaluation metrics in one call."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    return {
        "accuracy":   round(accuracy_score(y_test,  y_pred),                    4),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0),   4),
        "recall":     round(recall_score(y_test,    y_pred, zero_division=0),   4),
        "f1":         round(f1_score(y_test,        y_pred, zero_division=0),   4),
        "auc":        round(roc_auc_score(y_test, y_prob),                      4),
        "cm":         confusion_matrix(y_test, y_pred),
        "fpr":        fpr,
        "tpr":        tpr,
        "prec_curve": prec_c,
        "rec_curve":  rec_c,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  KMEANS SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def segment_customers(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    4-cluster KMeans segmentation on engagement + session behaviour + tenure.

    Appends two columns:
        segment        (int)  — cluster index
        segment_label  (str)  — human-readable label
    """
    feats = [
        "avg_engagement",
        "avg_duration",
        "total_sessions",
        "Tenure Months",
        "mobile_pct",
        "high_eng_pct",
    ]
    X_raw = df[feats].fillna(0)
    X_scaled = StandardScaler().fit_transform(X_raw)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["segment"] = km.fit_predict(X_scaled)

    # Label by mean engagement score, descending
    rank = (
        df.groupby("segment")["avg_engagement"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    labels = ["Champions", "Engaged", "At Risk", "Dormant"]
    df["segment_label"] = df["segment"].map(
        {seg: lbl for seg, lbl in zip(rank, labels)}
    )
    return df
