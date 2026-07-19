"""
Flask entry point.
"""

from pathlib import Path
from flask import Flask, render_template, request, jsonify

from pipeline.physicochemical import compute_physicochemical_features, classify_sequence
from pipeline.amp_activity import predict_amp_probability
from pipeline.toxicity import predict_toxicity
from pipeline.potency import predict_potency_class
from pipeline.decision import make_viability_verdict, Verdict
from pipeline.structure import submit_structure_job, poll_structure_job
from pipeline.validation import validate_sequence

BASE_DIR = Path(__file__).resolve().parent.parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Maps Verdict enum -> CSS class used for the colored banner
VERDICT_CSS_CLASS = {
    Verdict.VIABLE: "viable",
    Verdict.BORDERLINE: "borderline",
    Verdict.NOT_VIABLE: "not-viable",
}


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    sequence = request.form.get("sequence", "").strip().upper()

    is_valid, error_msg = validate_sequence(sequence)
    if not is_valid:
        return render_template("index.html", error=error_msg), 400

    features = compute_physicochemical_features(sequence)
    amp_result = predict_amp_probability(sequence, features)
    toxicity_result = predict_toxicity(sequence)
    potency_result = predict_potency_class(sequence, features)

    verdict = make_viability_verdict(
        amp_result=amp_result,
        toxicity_result=toxicity_result,
        potency_result=potency_result,
    )

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