"""
Checks the class balance of the AMP classifier's training data
(CAMPR3 + CAMPR4 positives vs. negatives.fasta) and warns if it's
skewed enough to bias the classifier.

A classifier trained on heavily imbalanced classes tends to just learn
"predict the majority class" as a shortcut -- e.g. with 17,000 positives
and 500 negatives, a model that says "AMP" for everything would already
be right 97% of the time on paper, without having learned anything real
about what makes a peptide antimicrobial.

Usage:
    PYTHONPATH=. python -m data.check_class_balance
"""

from data.amp_reference_loader import build_amp_classifier_dataset

# Ratios worse than this (majority:minority) are flagged as a real concern
WARNING_RATIO = 3.0


def main():
    df = build_amp_classifier_dataset()

    positive_count = (df["label"] == 1).sum()
    negative_count = (df["label"] == 0).sum()
    total = len(df)

    print(f"Total training examples: {total}")
    print(f"  Positives (AMP):     {positive_count}")
    print(f"  Negatives (non-AMP): {negative_count}")

    if negative_count == 0:
        print("\nERROR: zero negatives found. Check data/raw/negatives.fasta exists and has content.")
        return

    ratio = positive_count / negative_count
    print(f"\nPositive:Negative ratio = {ratio:.1f}:1")

    if ratio > WARNING_RATIO or ratio < (1 / WARNING_RATIO):
        majority = "positives" if ratio > 1 else "negatives"
        print(f"\nWARNING: class imbalance exceeds {WARNING_RATIO}:1 ({majority} dominate).")
        if ratio > 1:
            target_negatives = int(positive_count / WARNING_RATIO)
            print(f"Recommendation: regenerate negatives with a higher target count, e.g.:")
            print(f"  python data/generate_negatives.py --count {target_negatives}")
            print(f"(that would bring the ratio down to roughly {WARNING_RATIO}:1)")
        else:
            print("Recommendation: add more positive sequences (more CAMPR3/CAMPR4 data),")
            print("or reduce --count when regenerating negatives.")
    else:
        print(f"\nOK: ratio is within a reasonable range (under {WARNING_RATIO}:1).")


if __name__ == "__main__":
    main()