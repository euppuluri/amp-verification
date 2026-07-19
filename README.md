# AMP Clinical Viability Checker

Given a peptide sequence, predicts whether it's a reasonable candidate for
clinical testing by combining three independent checks:

1. **AMP likelihood** — is this actually an antimicrobial peptide?
2. **Toxicity** — is it likely toxic to human cells? (ToxinPred 3.0)
3. **Potency class** — Low / Medium / High, trained on GRAMPA + DBAASP MIC data

## Data sources

| Source          | Role                                                              | Where to get it |
|------------------|-------------------------------------------------------------------|------------------|
| GRAMPA           | MIC-labeled peptides — potency model training data                | https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/master/data/grampa.csv |
| DBAASP           | MIC-labeled peptides — potency model training data                | https://dbaasp.org/ (manual export) |
| CAMPR3 / CAMPR4  | Reference AMPs/non-AMPs — AMP classifier training data             | https://camp3.bicnirrh.res.in/ , https://camp.bicnirrh.res.in/ |
| APD3             | Reference natural AMPs — external sanity check                    | https://aps.unmc.edu/ |
| ToxinPred 3.0    | Toxicity score (external tool, not trained in-house)               | `pip install toxinpred3` |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Before the site can run

1. Get real data into `data/raw/` (see Data sources above), OR run
   `python data/generate_fake_data.py` to test with synthetic data first.
2. Train the models:
```bash
   PYTHONPATH=. python -m train.train_potency_model
   PYTHONPATH=. python -m train.train_amp_classifier
```
3. Start the site:
```bash
   PYTHONPATH=. python app/main.py
```
   Open http://127.0.0.1:5000

## Known constraints
- CAMPR3/CAMPR4/APD3 have no bulk API — export manually, refresh periodically.
- DBAASP's export column names vary by search settings — check
  `data/mic_data_loader.py`'s TODO once you've downloaded a real export.
- Verdict thresholds in `pipeline/decision.py` are starting points — tune
  once you have real validation data.