"""
Verifies the ToxinPred 3.0 integration actually works end-to-end against
your local installation -- this is the one piece of the pipeline that has
never been tested against a live install (my own sandbox has no internet,
so pipeline/toxicity.py's CLI flags and 'Hybrid Score' column name were
written from documentation, not confirmed against real output).

Tests against two known real peptides with very different expected
toxicity profiles:
  - Melittin: a well-characterized, genuinely toxic/hemolytic peptide
    (major component of bee venom) -- should score notably toxic.
  - LL-37: a human host-defense peptide, antimicrobial but not classified
    as a toxin -- should score non-toxic (some mild hemolytic activity at
    high concentrations is documented, but it should not be flagged the
    way melittin is).

If ToxinPred 3.0 isn't installed, or its output CSV doesn't have a
'Hybrid Score' / 'Prediction' column as pipeline/toxicity.py expects,
this will fail with a clear, specific error rather than a vague one --
read the printed message, it tells you exactly what to fix.

Usage:
    pip install toxinpred3
    PYTHONPATH=. python -m train.verify_toxinpred
"""

from pipeline.toxicity import predict_toxicity

TEST_CASES = {
    "Melittin (expected: toxic/high score)": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "LL-37 (expected: non-toxic/low score)": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
}


def main():
    print("Testing ToxinPred 3.0 integration against known peptides...\n")

    results = {}
    for label, sequence in TEST_CASES.items():
        print(f"--- {label} ---")
        print(f"Sequence: {sequence}")
        try:
            result = predict_toxicity(sequence)
            print(f"Toxicity score: {result.toxicity_score:.3f}")
            print(f"Prediction: {'TOXIC' if result.is_toxic else 'non-toxic'}")
            results[label] = result
        except FileNotFoundError:
            print("FAILED: 'toxinpred3' command not found.")
            print("Fix: pip install toxinpred3, and confirm it's on your PATH")
            print("     (try running 'toxinpred3 --help' directly in your terminal).")
            return
        except KeyError as e:
            print(f"FAILED: expected column {e} not found in ToxinPred's output CSV.")
            print("This means ToxinPred 3.0's actual output format differs from what")
            print("pipeline/toxicity.py assumes. Fix: run toxinpred3 manually on a test")
            print("FASTA file, open the output CSV, and check the real column names --")
            print("then update the column name in pipeline/toxicity.py to match.")
            return
        except Exception as e:
            print(f"FAILED with unexpected error: {e}")
            return
        print()

    print("=" * 50)
    print("Both predictions completed without errors.")
    print("Sanity check: melittin's score should be clearly higher than LL-37's.")
    melittin_result = results.get("Melittin (expected: toxic/high score)")
    ll37_result = results.get("LL-37 (expected: non-toxic/low score)")
    if melittin_result and ll37_result:
        if melittin_result.toxicity_score > ll37_result.toxicity_score:
            print("PASS: melittin scored higher than LL-37, as expected.")
        else:
            print("UNEXPECTED: melittin did not score higher than LL-37.")
            print("The integration is technically working (no errors), but the")
            print("actual scores don't match known biology -- worth double-checking")
            print("which ToxinPred model/threshold you're running (-m and -t flags).")


if __name__ == "__main__":
    main()