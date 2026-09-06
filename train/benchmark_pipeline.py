"""
Benchmarking suite that directly addresses the reviewer feedback:

  1. AUROC / AUPRC / F1 / MCC of the integrated pipeline vs. simpler
     individual predictors, on the SAME curated AMP/non-AMP held-out set.
  2. Whether combining features into one model improves ROBUSTNESS, not
     just a single point-estimate metric -- tested via bootstrap
     resampling of the held-out test set (1000 resamples -> 95% CIs).
  3. Ablation: which individual features drive the integrated model's
     AUROC. This is an AUROC-based leave-one-feature-out analysis; it
     complements train/ablation_study.py's accuracy-based permutation
     importance with the metric this benchmark (and a manuscript
     reviewer) actually cares about.

The two "individual predictors" being benchmarked against are:
  - A charge-based heuristic (cationicity screening is the oldest,
    simplest published rule for flagging candidate AMPs).
  - The single best-performing physicochemical feature on its own,
    chosen by TRAIN-set AUROC (never test-set) so we aren't
    cherry-picking after the fact. This is the fairest possible
    single-tool baseline -- if the integrated model can't beat this,
    integration isn't adding real value.

Usage:
    PYTHONPATH=. python -m train.benchmark_pipeline

Outputs (written to train/benchmark_results/):
    metrics_table.csv    -- point estimates + 95% bootstrap CI per model
    ablation_table.csv   -- AUROC drop when each feature is removed
    roc_curves.png       -- overlaid ROC curves, ready for a manuscript figure
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    matthews_corrcoef, roc_curve,
)

from data.amp_reference_loader import build_amp_classifier_dataset
from pipeline.physicochemical import compute_physicochemical_features
from train.homology_split import homology_partition_split

# sklearn's LogisticRegression can throw benign convergence warnings on
# single-feature fits -- suppressing them keeps the benchmark output readable.
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "length", "net_charge", "hydrophobicity", "hydrophobic_moment",
    "helicity_score", "aggregation_propensity", "isoelectric_point", "protease_stability",
]

OUT_DIR = Path(__file__).parent / "benchmark_results"
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def build_dataset() -> pd.DataFrame:
    df = build_amp_classifier_dataset()
    rows, kept = [], []
    for row in df.itertuples(index=False):
        try:
            feats = compute_physicochemical_features(row.sequence)
        except KeyError:
            continue  # non-standard residue (X, B, Z, U, O...) -- skip, don't crash
        rows.append(feats.as_dict())
        kept.append(row)
    df = pd.DataFrame(kept).reset_index(drop=True)
    feat_df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        df[c] = feat_df[c]
    return df


def fit_charge_heuristic(train: pd.DataFrame):
    """Classic cationicity-screening rule. Threshold tuned on TRAIN only
    (by MCC), exactly like the sklearn models below are tuned -- so this
    baseline gets the same fairness treatment, not a strawman default."""
    best_threshold, best_mcc = None, -1
    lo, hi = train["net_charge"].min(), train["net_charge"].max()
    for t in np.arange(lo, hi, 0.1):
        pred = (train["net_charge"] >= t).astype(int)
        mcc = matthews_corrcoef(train["label"], pred)
        if mcc > best_mcc:
            best_mcc, best_threshold = mcc, t
    return best_threshold


def predict_charge_heuristic(df: pd.DataFrame, threshold: float):
    """Min-max-scaled charge stands in for a 'probability' so this
    rule-based baseline can still be scored with AUROC/AUPRC -- a fair way
    to compare a non-probabilistic screening rule against real classifiers."""
    charge = df["net_charge"]
    span = charge.max() - charge.min()
    proba = (charge - charge.min()) / (span if span > 0 else 1.0)
    pred = (charge >= threshold).astype(int)
    return proba.values, pred.values


def fit_best_single_feature(train: pd.DataFrame):
    """Picks whichever single physicochemical feature predicts AMP status
    best on its own (by TRAIN AUROC), as a logistic regression. This is
    the strongest possible 'one existing signal' baseline."""
    best_col, best_auroc, best_model = None, -1, None
    for col in FEATURE_COLS:
        model = LogisticRegression(max_iter=1000)
        model.fit(train[[col]], train["label"])
        proba = model.predict_proba(train[[col]])[:, 1]
        auroc = roc_auc_score(train["label"], proba)
        if auroc > best_auroc:
            best_auroc, best_col, best_model = auroc, col, model
    return best_col, best_model


def point_metrics(y_true, y_proba, y_pred) -> dict:
    return {
        "AUROC": roc_auc_score(y_true, y_proba),
        "AUPRC": average_precision_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_proba, y_pred, n=N_BOOTSTRAP):
    """95% CI per metric via bootstrap resampling of the held-out test
    set. This is the actual robustness check: a model whose CI doesn't
    overlap the baselines' CI is a genuinely robust improvement, not a
    lucky point estimate on one split."""
    y_true, y_proba, y_pred = np.asarray(y_true), np.asarray(y_proba), np.asarray(y_pred)
    n_samples = len(y_true)
    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for _ in range(n):
        idx = rng.randint(0, n_samples, n_samples)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUROC/MCC undefined with only one class in this resample
        try:
            rows.append({
                "AUROC": roc_auc_score(yt, y_proba[idx]),
                "AUPRC": average_precision_score(yt, y_proba[idx]),
                "F1": f1_score(yt, y_pred[idx]),
                "MCC": matthews_corrcoef(yt, y_pred[idx]),
            })
        except ValueError:
            continue
    boot_df = pd.DataFrame(rows)
    return {m: (boot_df[m].quantile(0.025), boot_df[m].quantile(0.975)) for m in boot_df.columns}


def run_ablation(train, test, full_auroc):
    """Leave-one-feature-out: retrain the integrated model without each
    feature in turn and measure the AUROC drop. Bigger drop = that
    feature contributes more to the integrated model's real advantage
    over single-feature baselines."""
    rows = []
    for dropped in FEATURE_COLS:
        cols = [c for c in FEATURE_COLS if c != dropped]
        model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(train[cols], train["label"])
        proba = model.predict_proba(test[cols])[:, 1]
        auroc = roc_auc_score(test["label"], proba)
        rows.append({
            "feature_removed": dropped,
            "auroc_without_feature": round(auroc, 4),
            "auroc_drop": round(full_auroc - auroc, 4),
        })
    return pd.DataFrame(rows).sort_values("auroc_drop", ascending=False).reset_index(drop=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = build_dataset()
    train, test = homology_partition_split(df)
    print(f"Train: {len(train)}, Test: {len(test)} (homology-partitioned, no leakage)\n")

    models = {}  # name -> (proba, pred)

    threshold = fit_charge_heuristic(train)
    models[f"Charge heuristic (net_charge >= {threshold:.2f})"] = predict_charge_heuristic(test, threshold)

    best_col, single_model = fit_best_single_feature(train)
    single_proba = single_model.predict_proba(test[[best_col]])[:, 1]
    single_pred = single_model.predict(test[[best_col]])
    models[f"Best single feature ({best_col})"] = (single_proba, single_pred)

    full_model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    full_model.fit(train[FEATURE_COLS], train["label"])
    full_proba = full_model.predict_proba(test[FEATURE_COLS])[:, 1]
    full_pred = full_model.predict(test[FEATURE_COLS])
    models["Integrated pipeline (all 8 features)"] = (full_proba, full_pred)

    print(f"{'Model':42s} {'AUROC (95% CI)':>24s} {'AUPRC':>10s} {'F1':>8s} {'MCC':>8s}")
    print("-" * 96)

    summary_rows = []
    for name, (proba, pred) in models.items():
        point = point_metrics(test["label"], proba, pred)
        ci = bootstrap_ci(test["label"], proba, pred)
        auroc_ci_str = f"{point['AUROC']:.3f} [{ci['AUROC'][0]:.3f}-{ci['AUROC'][1]:.3f}]"
        print(f"{name:42s} {auroc_ci_str:>24s} {point['AUPRC']:>10.3f} {point['F1']:>8.3f} {point['MCC']:>8.3f}")
        summary_rows.append({
            "model": name,
            **{k: round(v, 4) for k, v in point.items()},
            **{f"{k}_ci_low": round(v[0], 4) for k, v in ci.items()},
            **{f"{k}_ci_high": round(v[1], 4) for k, v in ci.items()},
        })

    metrics_df = pd.DataFrame(summary_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_table.csv", index=False)

    full_auroc = point_metrics(test["label"], full_proba, full_pred)["AUROC"]
    ablation_df = run_ablation(train, test, full_auroc)
    ablation_df.to_csv(OUT_DIR / "ablation_table.csv", index=False)

    print("\nAblation (AUROC drop when each feature is removed from the integrated model):")
    print(ablation_df.to_string(index=False))

    # --- ROC curve figure, for direct use in a manuscript ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        for name, (proba, _) in models.items():
            fpr, tpr, _ = roc_curve(test["label"], proba)
            auroc = roc_auc_score(test["label"], proba)
            plt.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC: integrated pipeline vs. individual predictors")
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "roc_curves.png", dpi=150)
        print(f"\nSaved ROC curve figure to {OUT_DIR / 'roc_curves.png'}")
    except ImportError:
        print("\n(matplotlib not installed -- skipping roc_curves.png; "
              "metrics_table.csv and ablation_table.csv were still written.)")

    print(f"\nWrote {OUT_DIR / 'metrics_table.csv'} and {OUT_DIR / 'ablation_table.csv'}")
    print("\nHow to read this for the manuscript:")
    print("- If the integrated pipeline's AUROC 95% CI does not overlap the")
    print("  charge-heuristic's or best-single-feature's CI, that's a robust,")
    print("  not just lucky, improvement from combining features.")
    print("- The ablation table tells you which features are doing the real")
    print("  work -- report the top 2-3 as your mechanistic explanation for")
    print("  WHY integration helps, not just THAT it helps.")


if __name__ == "__main__":
    main()
