"""
Case study: takes a Low/Medium-potency candidate peptide, scored against
the Gram-positive potency model (relevant to S. aureus, a Gram-positive
target), and demonstrates the optimization engine (pipeline/optimization.py)
proposing concrete improvements -- addressing the mentor's request for
"a case study using your previously designed anti-S. aureus peptide
candidates."

This produces a before/after report suitable for a manuscript figure or
table: original sequence -> physicochemical properties -> predicted
potency/AMP-probability, then the top suggested variants with their
predicted improvements and the specific rationale for each mutation.

Usage:
    PYTHONPATH=. python -m train.run_case_study "YOUR_SEQUENCE_HERE"
    (or with no argument, uses a built-in example sequence)
"""

import sys

from pipeline.physicochemical import compute_physicochemical_features
from pipeline.amp_activity import predict_amp_probability
from pipeline.potency import predict_potency_class
from pipeline.optimization import suggest_improvements

# A plausible example: cationic, moderate length, but not obviously
# high-potency -- representative of the "Low/Medium" case this feature
# is designed for. Swap for a real candidate from your own screening.
EXAMPLE_SEQUENCE = "GLLDIVKKVVGALGSL"


def print_peptide_report(label, sequence, variant):
    features = compute_physicochemical_features(sequence)
    amp_result = predict_amp_probability(sequence, features)
    potency_result = predict_potency_class(sequence, features, variant=variant)

    print(f"\n{label}: {sequence}")
    print(f"  Length: {features.length}  |  Net charge: {features.net_charge:+.2f}  |  "
          f"Hydrophobicity: {features.hydrophobicity:+.2f}")
    print(f"  Hydrophobic moment: {features.hydrophobic_moment:.3f}  |  "
          f"Helicity: {features.helicity_score:.2f}  |  Protease stability: {features.protease_stability:.2f}")
    print(f"  AMP probability: {amp_result.probability:.3f}")
    print(f"  Potency class: {potency_result.potency_class.value}  "
          f"(model: {potency_result.model_variant_used})")
    print(f"  Class probabilities: {', '.join(f'{k}={v:.2f}' for k, v in potency_result.class_probabilities.items())}")

    return potency_result.potency_class.value


def main():
    sequence = sys.argv[1].upper() if len(sys.argv) > 1 else EXAMPLE_SEQUENCE
    variant = "gram_positive"  # S. aureus is Gram-positive

    print("=" * 70)
    print("CASE STUDY: Peptide optimization for a Gram-positive (S. aureus) target")
    print("=" * 70)

    original_class = print_peptide_report("ORIGINAL SEQUENCE", sequence, variant)

    if original_class == "High":
        print("\nThis sequence already predicts High potency -- optimization")
        print("suggestions are only generated for Low/Medium starting points.")
        print("Try a different sequence, or remove this check to see suggestions anyway.")
        return

    print("\n" + "=" * 70)
    print("GENERATING OPTIMIZATION SUGGESTIONS...")
    print("=" * 70)

    suggestions = suggest_improvements(sequence, variant=variant, num_suggestions=8)

    if not suggestions:
        print("No improving candidates found among generated variants.")
        return

    print(f"\nTop {len(suggestions)} suggested variants (ranked by predicted improvement):\n")
    for i, s in enumerate(suggestions, 1):
        print(f"--- Suggestion {i} ---")
        print(f"  Sequence: {s.sequence}")
        print(f"  Change: {s.mutation_description}")
        print(f"  Potency: {s.original_potency_class} -> {s.new_potency_class}")
        print(f"  AMP probability: {s.new_amp_probability:.3f}")
        print(f"  Key feature changes: " + ", ".join(
            f"{k}={v:+.2f}" for k, v in s.feature_deltas.items() if abs(v) > 0.01
        ))
        print(f"  Improvement score: {s.improvement_score:.3f}")
        print()

    print("=" * 70)
    print("IMPORTANT CAVEAT FOR ANY MANUSCRIPT USE OF THIS CASE STUDY:")
    print("=" * 70)
    print("These suggestions are computational predictions from the trained")
    print("models, NOT experimentally validated improvements. Toxicity was")
    print("NOT checked for these candidates (ToxinPred wasn't run per-variant")
    print("for speed) -- any candidate taken forward for real synthesis should")
    print("be re-screened individually through the full pipeline, including")
    print("live toxicity prediction, before being treated as a real lead.")


if __name__ == "__main__":
    main()