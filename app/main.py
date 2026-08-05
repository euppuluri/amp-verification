"""
Flask entry point.
"""

from pathlib import Path
from flask import Flask, render_template, request, jsonify

from pipeline.physicochemical import compute_physicochemical_features, classify_sequence
from pipeline.amp_activity import predict_amp_probability
from pipeline.toxicity import predict_toxicity
from pipeline.potency import predict_potency_class, available_variants, VARIANT_LABELS
from pipeline.decision import make_viability_verdict, Verdict
from pipeline.candidacy import compute_candidacy_score
from pipeline.structure import submit_structure_job, poll_structure_job
from pipeline.validation import validate_sequence

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

VERDICT_CSS_CLASS = {
    Verdict.VIABLE: "viable",
    Verdict.BORDERLINE: "borderline",
    Verdict.NOT_VIABLE: "not-viable",
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        variant_options=VARIANT_LABELS,
        variant_availability=available_variants(),
    )


@app.route("/predict", methods=["POST"])
def predict():
    sequence = request.form.get("sequence", "").strip().upper()
    requested_variant = request.form.get("gram_context", "pooled")

    is_valid, error_msg = validate_sequence(sequence)
    if not is_valid:
        return render_template(
            "index.html", error=error_msg,
            variant_options=VARIANT_LABELS, variant_availability=available_variants(),
        ), 400

    try:
        features = compute_physicochemical_features(sequence)
    except KeyError as e:
        return render_template(
            "index.html",
            error=f"This sequence contains a residue ({e}) our feature formulas don't cover "
                  f"(only the 20 standard amino acids are supported).",
            variant_options=VARIANT_LABELS, variant_availability=available_variants(),
        ), 400

    try:
        amp_result = predict_amp_probability(sequence, features)
        # fallback_to_pooled=True means: if you asked for a Gram-specific
        # model that hasn't been trained yet, silently use the pooled model
        # instead rather than failing outright. The results page tells you
        # if that happened.
        potency_result = predict_potency_class(sequence, features, variant=requested_variant)
    except FileNotFoundError:
        return render_template(
            "index.html",
            error="Model files are missing. Run 'PYTHONPATH=. python -m train.train_amp_classifier' "
                  "and 'PYTHONPATH=. python -m train.train_potency_model' first, then retry.",
            variant_options=VARIANT_LABELS, variant_availability=available_variants(),
        ), 500

    try:
        toxicity_result = predict_toxicity(sequence)
    except FileNotFoundError:
        return render_template(
            "index.html",
            error="ToxinPred 3.0 isn't installed or isn't on your PATH. "
                  "Run 'pip install toxinpred3' and retry.",
            variant_options=VARIANT_LABELS, variant_availability=available_variants(),
        ), 500

    verdict = make_viability_verdict(
        amp_result=amp_result,
        toxicity_result=toxicity_result,
        potency_result=potency_result,
    )

    candidacy = compute_candidacy_score(
        amp_result=amp_result,
        toxicity_result=toxicity_result,
        potency_result=potency_result,
        protease_stability=features.protease_stability,
    )

    fell_back = (requested_variant != potency_result.model_variant_used)

    return render_template(
        "results.html",
        sequence=sequence,
        residue_tiles=classify_sequence(sequence),
        features=features,
        amp_result=amp_result,
        toxicity_result=toxicity_result,
        potency_result=potency_result,
        verdict=verdict,
        verdict_css_class=VERDICT_CSS_CLASS[verdict.verdict],
        candidacy=candidacy,
        requested_variant_label=VARIANT_LABELS.get(requested_variant, requested_variant),
        used_variant_label=VARIANT_LABELS.get(potency_result.model_variant_used, potency_result.model_variant_used),
        fell_back=fell_back,
    )


@app.route("/predict/structure", methods=["POST"])
def predict_structure():
    sequence = request.form.get("sequence", "").strip().upper()
    is_valid, error_msg = validate_sequence(sequence)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    job_id = submit_structure_job(sequence)
    return jsonify({"job_id": job_id, "status": "submitted"})


@app.route("/predict/structure/<job_id>", methods=["GET"])
def structure_status(job_id):
    result = poll_structure_job(job_id)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)