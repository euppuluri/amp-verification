"""
Optional peptide-shape prediction via AlphaFold3 Server.

AlphaFold3 has no locally downloadable weights — jobs must be submitted
to DeepMind's hosted AlphaFold Server and polled, which can take minutes
to hours. That's why this is a separate async endpoint, not part of the
synchronous /predict flow — and NOT used in the viability verdict itself.
"""

import uuid

_JOBS: dict[str, dict] = {}


def submit_structure_job(sequence: str) -> str:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {"status": "submitted", "sequence": sequence, "result": None}

    # TODO: replace with real AlphaFold3 Server submission call

    return job_id


def poll_structure_job(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        return {"status": "not_found"}

    # TODO: query AlphaFold3 Server for job status

    return job