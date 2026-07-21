"""
External validation: tests the trained AMP classifier against APD3 —
a database it was NEVER trained on (see data/amp_reference_loader.py).

This checks whether the model generalizes to an independent set of known
antimicrobial peptides, rather than just fitting patterns specific to
CAMPR3/CAMPR4. Since APD3 contains only validated AMPs (no negatives),
the metric here is recall/sensitivity: what fraction of these known AMPs
does the model correctly flag as AMP?

A high training-set accuracy paired with a much lower APD3 recall would
indicate overfitting to CAMPR-specific sequence patterns. A comparable
number is a genuinely encouraging sign.

Run after train_amp_classifier.py has produced models/amp_classifier.pkl:
    PYTHONPATH=. python -m train.validate_external
"""

from data.amp_reference_loader import build_apd3_validation_set
from pipeline.physicochemical import compute_physicochemical_features
from pipeline.amp_activity import predict_amp_probability


def main():
    apd3 = build_apd3_validation_set()
    print(f"Loaded {len(apd3)} APD3 peptides for external validation.\n")

    correct = 0
    probabilities = []

    for seq in apd3["sequence"]:
        features = compute_physicochemical_features(seq)
        result = predict_amp_probability(seq, features)
        probabilities.append(result.probability)
        if result.is_amp:
            correct += 1

    recall = correct / len(apd3)
    mean_probability = sum(probabilities) / len(probabilities)

    print(f"APD3 external validation recall (sensitivity): {recall:.3f}")
    print(f"  ({correct} / {len(apd3)} known AMPs correctly flagged as AMP)")
    print(f"Mean predicted AMP probability across APD3: {mean_probability:.3f}")
    print()
    print("For context: compare this recall against the training-set CV")
    print("accuracy printed by train_amp_classifier.py. A big gap between")
    print("them suggests the model is overfitting to CAMPR-specific patterns")
    print("rather than learning generalizable AMP features.")


if __name__ == "__main__":
    main()