"""
Combined case study: comprehensive pipeline validation AND optimization guidance.

PART 1 (Validation): Score your designed anti-S. aureus candidate against
a panel of literature-known AMPs, demonstrating that the integrated pipeline
correctly prioritizes candidates. Shows ranking and candidacy score.

PART 2 (Optimization): If your candidate is Low or Medium potency, generate
and score concrete sequence variants predicted to improve it, using the
ACTUAL trained AMP and potency classifiers (not simulated scoring).
Produces before/after improvements with explainable rationale for each
mutation.

Together, this addresses both:
- The reviewer's request for a case study validating the pipeline
- The mentor's vision of optimization-guided design feedback

Usage:
    PYTHONPATH=. python -m train.run_case_study_combined
"""

import sys
from pathlib import Path

import pandas as pd

from pipeline.physicochemical import compute_physicochemical_features
from pipeline.amp_activity import predict_amp_probability
from pipeline.potency import predict_potency_class
from pipeline.candidacy import compute_candidacy_score
from pipeline.decision import make_viability_verdict
from pipeline.optimization import suggest_improvements

try:
    from pipeline.toxicity import predict_toxicity
    TOXINPRED_AVAILABLE = True
except ImportError:
    TOXINPRED_AVAILABLE = False

OUT_DIR = Path(__file__).parent / "benchmark_results"

YOUR_CANDIDATE = "IKLRVKWRIKLRVKWREKLRIKWR"

REFERENCE_PEPTIDES = {
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin_2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Pexiganan_MSI-78": "GIGKFLKKAKKFGKAFVKILKK",
    "Aurein_1.2": "GLFDIIKKIAESF",
    "Temporin_A": "FLPLIGRVLSGIL",
    "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
}


class _NullToxicityResult:
    toxicity_score = float("nan")
    is_toxic = None


def score_peptide(sequence, name, source="Your design"):
    sequence = sequence.upper()
    features = compute_physicochemical_features(sequence)
    amp_result = predict_amp_probability(sequence, features)
    potency_result = predict_potency_class(sequence, features, variant="gram_positive")

    if TOXINPRED_AVAILABLE:
        try:
            toxicity_result = predict_toxicity(sequence)
        except Exception as e:
            toxicity_result = _NullToxicityResult()
    else:
        toxicity_result = _NullToxicityResult()

    candidacy = compute_candidacy_score(
        amp_result=amp_result,
        toxicity_result=toxicity_result,
        potency_result=potency_result,
        protease_stability=features.protease_stability,
    )
    verdict = make_viability_verdict(amp_result, toxicity_result, potency_result)

    single_tool_verdict = "Promising" if amp_result.probability >= 0.7 else "Not promising"

    return {
        "peptide": name,
        "source": source,
        "sequence": sequence,
        "length": features.length,
        "net_charge": round(features.net_charge, 2),
        "hydrophobicity": round(features.hydrophobicity, 3),
        "hydrophobic_moment": round(features.hydrophobic_moment, 3),
        "helicity_score": round(features.helicity_score, 3),
        "protease_stability": round(features.protease_stability, 3),
        "amp_probability": round(amp_result.probability, 3),
        "toxicity_score": (round(toxicity_result.toxicity_score, 3)
                            if toxicity_result.toxicity_score == toxicity_result.toxicity_score
                            else "n/a"),
        "potency_class": potency_result.potency_class.value,
        "candidacy_score": round(candidacy.score, 3),
        "single_tool_verdict": single_tool_verdict,
        "integrated_verdict": verdict.verdict.value,
    }


def main():
    if not TOXINPRED_AVAILABLE:
        print("NOTE: ToxinPred 3.0 isn't importable -- toxicity columns will show 'n/a'.\n")

    candidate = sys.argv[1].upper() if len(sys.argv) > 1 else YOUR_CANDIDATE

    print("=" * 90)
    print("CASE STUDY: Designed Anti-S. aureus Peptide + Optimization Suggestions")
    print("=" * 90)

    print("\nPART 1: VALIDATION")
    print("-" * 90)
    print("Scoring your designed candidate against known anti-S. aureus peptides.\n")

    results = []
    your_result = score_peptide(candidate, "Your_Candidate", source="Your design")
    results.append(your_result)

    for name, seq in REFERENCE_PEPTIDES.items():
        ref_result = score_peptide(seq, name, source="Reference (literature)")
        results.append(ref_result)

    df = pd.DataFrame(results)
    df_sorted = df.sort_values("candidacy_score", ascending=False).reset_index(drop=True)

    display_cols = ["peptide", "potency_class", "amp_probability", "candidacy_score", "integrated_verdict"]
    print(df_sorted[display_cols].to_string(index=False))
    print()

    your_rank = df_sorted[df_sorted["source"] == "Your design"].index[0] + 1
    your_candidacy = df_sorted[df_sorted["source"] == "Your design"]["candidacy_score"].iloc[0]
    your_potency = df_sorted[df_sorted["source"] == "Your design"]["potency_class"].iloc[0]

    print(f"\n→ Your candidate ranked #{your_rank} of {len(df_sorted)} by candidacy score ({your_candidacy}).")
    print(f"→ Potency class (Gram-positive target): {your_potency}")

    if your_potency == "High":
        print("\n" + "=" * 90)
        print("PART 2: OPTIMIZATION")
        print("-" * 90)
        print(f"Your candidate already predicts High potency -- no improvements needed.")
    else:
        print("\n" + "=" * 90)
        print("PART 2: OPTIMIZATION")
        print("-" * 90)
        print(f"Your candidate predicts {your_potency} potency. Generating improvement suggestions...\n")

        suggestions = suggest_improvements(candidate, variant="gram_positive", num_suggestions=5)

        if not suggestions:
            print("No improving candidates found among generated variants.")
        else:
            print(f"Top {len(suggestions)} suggested variants (ranked by predicted improvement):\n")

            improvement_data = []
            for i, s in enumerate(suggestions, 1):
                print(f"{'─' * 90}")
                print(f"SUGGESTION {i}")
                print(f"{'─' * 90}")
                print(f"Sequence: {s.sequence}")
                print(f"Change:   {s.mutation_description}")
                print(f"Potency:  {s.original_potency_class} → {s.new_potency_class}")
                print(f"AMP probability: {s.new_amp_probability:.3f}")

                key_deltas = {k: v for k, v in s.feature_deltas.items() if abs(v) > 0.01 and k != "length"}
                if key_deltas:
                    print(f"Key feature changes: {', '.join(f'{k}={v:+.3f}' for k, v in key_deltas.items())}")

                print(f"Improvement score: {s.improvement_score:.3f}")
                print()

                improvement_data.append({
                    "suggestion_rank": i,
                    "sequence": s.sequence,
                    "mutation": s.mutation_description,
                    "original_potency": s.original_potency_class,
                    "new_potency": s.new_potency_class,
                    "new_amp_probability": round(s.new_amp_probability, 3),
                    "improvement_score": s.improvement_score,
                })

            improvement_df = pd.DataFrame(improvement_data)
            OUT_DIR.mkdir(exist_ok=True)
            improvement_csv = OUT_DIR / "optimization_suggestions.csv"
            improvement_df.to_csv(improvement_csv, index=False)
            print(f"\nSuggestions saved to {improvement_csv}")

    OUT_DIR.mkdir(exist_ok=True)
    validation_csv = OUT_DIR / "case_study_validation.csv"
    df_sorted.to_csv(validation_csv, index=False)
    print(f"\nValidation results saved to {validation_csv}")

    print("\n" + "=" * 90)
    print("IMPORTANT NOTES FOR MANUSCRIPT SUBMISSION")
    print("=" * 90)
    print("PART 1 (Validation): Your candidate scored with real trained models.")
    print("PART 2 (Optimization): Computational predictions only -- NOT experimentally")
    print("validated. Toxicity NOT checked. Suggested variants require full re-screening")
    print("before any real synthesis consideration.")


if __name__ == "__main__":
    main()
