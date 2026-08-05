# AMP Clinical Viability Checker

Given a peptide sequence, predicts whether it's a reasonable candidate for
clinical testing by combining:

1. **AMP likelihood** — is this actually an antimicrobial peptide?
2. **Toxicity** — is it likely toxic to human cells? (ToxinPred 3.0)
3. **Potency class** — Low / Medium / High, trained on GRAMPA + DBAASP MIC
   data, optionally Gram-stratified
4. **Candidacy score** — a transparent weighted composite of the above,
   plus protease stability

## Data sources

| Source          | Role                                                 | Where to get it |
|------------------|-------------------------------------------------------|------------------|
| GRAMPA           | MIC-labeled peptides — potency model training data     | https://github.com/zswitten/Antimicrobial-Peptides/raw/refs/heads/master/data/grampa.csv |
| DBAASP           | MIC-labeled peptides — potency model training data     | via `data/fetch_dbaasp.py` (DBAASP's REST API) |
| CAMPR3 / CAMPR4  | Reference AMPs — AMP classifier training data          | https://camp3.bicnirrh.res.in/ , https://camp.bicnirrh.res.in/ |
| APD3             | Reference AMPs — held out entirely for external validation, not used in training | https://aps.unmc.edu/ |
| ToxinPred 3.0    | Toxicity score (external tool)                         | `pip install toxinpred3` |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Before the site can run

1. Get data into `data/raw/`:
   - Real: download GRAMPA (link above), run `data/fetch_dbaasp.py` for DBAASP,
     manually export CAMPR3/CAMPR4/APD3.
   - Or test with fake data first: `python data/generate_fake_data.py`
2. Train the models:
```bash
   PYTHONPATH=. python -m train.train_amp_classifier
   PYTHONPATH=. python -m train.train_potency_model
   # Optional Gram-stratified variants:
   PYTHONPATH=. python -m train.train_potency_model --gram negative
   PYTHONPATH=. python -m train.train_potency_model --gram positive
```
3. (Optional) External validation:
```bash
   PYTHONPATH=. python -m train.validate_external
```
4. Start the site:
```bash
   PYTHONPATH=. python app/main.py
```
   Open http://127.0.0.1:5000

## Architecture