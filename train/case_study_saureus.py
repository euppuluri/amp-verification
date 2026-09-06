"""
Case study: runs your designed anti-S. aureus candidate(s) through the
FULL integrated pipeline (AMP probability -> toxicity -> Gram-positive
potency class -> candidacy score -> viability verdict), alongside a small
panel of well-characterized literature AMPs for context, and contrasts
all of it with what a single-tool workflow (AMP-probability screening
alone) would have concluded.

Directly addresses the reviewer's request for "a case study using your
previously designed anti-S. aureus peptide candidates."

YOUR_CANDIDATES holds the peptide(s) you actually designed. Add more
name -> sequence pairs here as you design them -- nothing else in the
script needs to change.

REFERENCE_PEPTIDES is NOT part of your case study result -- it exists
only so your one designed candidate has known-quantity peptides to be
ranked against, rather than being scored in isolation with no sense of
what a "good" or "bad" candidacy score even looks like. It deliberately
includes melittin, a highly active, highly cationic/amphipathic AMP that
is ALSO well-documented as hemolytic/toxic to human cells, and LL-37, a
human-derived AMP with essentially no toxicity to human cells -- between
them they bracket the range your candidate's own score should be read
against.

Usage:
    PYTHONPATH=. python -m train.case_study_saureus
"""

from pathlib import Path

import pandas as pd

from pipeline.physicochemical import compute_physicochemical_features
from pipeline.amp_activity import predict_amp_probability
from pipeline.potency import predict_potency_class
from pipeline.candidacy import compute_candidacy_score
from pipeline.decision import make_viability_verdict

try:
    from pipeline.toxicity import predict_toxicity
    TOXINPRED_AVAILABLE = True
except ImportError:
    TOXINPRED_AVAILABLE = False

OUT_DIR = Path(__file__).parent / "benchmark_results"

# Your actual designed candidate(s). Add more entries as you design them.
YOUR_CANDIDATES = {
    "Designed_1": "IKLRVKWRIKLRVKWREKLRIKWR",
}

# Literature-documented peptides with anti-S. aureus activity, included
# ONLY for ranking context -- see module docstring. Sources:
# well-established AMP literature / APD3 entries for each named peptide.
REFERENCE_PEPTIDES = {
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin_2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Pexiganan_MSI-78": "GIGKFLKKAKKFGKAFVKILKK",
    "Aurein_1.2": "GLFDIIKKIAESF",
    "Temporin_A": "FLPLIGRVLSGIL",
    "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
}


class _NullToxicityResult:
    """Used only if ToxinPred 3.0 isn't installed, so the case study can
    still run and show every OTHER stage of the pipeline. The printed
    table clearly marks these rows as toxicity-unscored."""
    toxicity_score = float("nan")
    is_toxic = None


def run_case_study():
    rows = []
    all_peptides = (
        [(name, seq, "Your design") for name, seq in YOUR_CANDIDATES.items()]
        + [(name, seq, "Reference (literature)") for name, seq in REFERENCE_PEPTIDES.items()]
    )
    for name, sequence, source in all_peptides:
        sequence = sequence.upper()
        features = compute_physicochemical_features(sequence)
        amp_result = predict_amp_probability(sequence, features)
        potency_result = predict_potency_class(sequence, features, variant="gram_positive")

        if TOXINPRED_AVAILABLE:
            try:
                toxicity_result = predict_toxicity(sequence)
            except Exception as e:
                print(f"  (toxicity prediction failed for {name}: {e} -- treating as unscored)")
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

        # What a single-tool (AMP-probability-only) workflow would have
        # concluded, for direct comparison against the integrated verdict.
        single_tool_verdict = "Promising" if amp_result.probability >= 0.7 else "Not promising"

        rows.append({
            "peptide": name,
            "source": source,
            "length": features.length,
            "net_charge": features.net_charge,
            "amp_probability": round(amp_result.probability, 3),
            "toxicity_score": (round(toxicity_result.toxicity_score, 3)
                                if toxicity_result.toxicity_score == toxicity_result.toxicity_score
                                else "n/a"),  # NaN check without importing math/numpy
            "potency_class (Gram+)": potency_result.potency_class.value,
            "candidacy_score": candidacy.score,
            "single_tool_verdict (AMP prob. only)": single_tool_verdict,
            "integrated_verdict": verdict.verdict.value,
        })

    df = pd.DataFrame(rows)
    return df.sort_values("candidacy_score", ascending=False).reset_index(drop=True)


def main():
    if not TOXINPRED_AVAILABLE:
        print("NOTE: ToxinPred 3.0 isn't importable in this environment -- toxicity "
              "columns will show 'n/a' and the integrated verdict will rely only on "
              "AMP probability + potency for this run. Install it for the real "
              "manuscript numbers (see pipeline/toxicity.py docstring).\n")

    results = run_case_study()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "case_study_saureus.csv"
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nWrote {out_path}")

    your_rows = results[results["source"] == "Your design"]
    rank = results.index[results["source"] == "Your design"].tolist()

    print("\nHow to read this for the manuscript:")
    print("- Compare the 'single_tool_verdict' column (AMP probability alone)")
    print("  against 'integrated_verdict'. Any row where they DISAGREE is your")
    print("  concrete evidence that integration changes real decisions, not just")
    print("  aggregate benchmark metrics.")
    print("- The reference panel is context, not part of your result: melittin")
    print("  (active but toxic) and LL-37 (non-toxic, human-derived) bracket the")
    print("  candidacy-score range your own design should be read against.")
    if not your_rows.empty:
        for i, r in your_rows.iterrows():
            print(f"- Your candidate '{r['peptide']}' ranked #{rank[list(your_rows.index).index(i)] + 1} "
                  f"of {len(results)} by candidacy score ({r['candidacy_score']:.3f}), "
                  f"verdict: {r['integrated_verdict']}.")
    print("- Add more of your own designed sequences to YOUR_CANDIDATES as you")
    print("  make them -- a case study with 3-5 of your own candidates against")
    print("  this same reference panel will read stronger in the manuscript than")
    print("  a single design point.")


if __name__ == "__main__":
    main()
