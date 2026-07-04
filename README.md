# Fairness in Credit-Risk Classification

**Group members:** Davis Joseph and Malone Mfono  
**Institution:** Aivancity  
**Professor:** Professor David Appadourai  
**Repository:** https://github.com/davisjoseph6/fairness-lab.git

## Project objective

This project audits and mitigates fairness disparities in a binary
credit-risk classifier trained on the German Credit dataset from OpenML.

The sensitive attribute is **age**, divided into two groups:

- **Young:** age < 25
- **Old:** age >= 25

The prediction target is:

- **1:** good credit
- **0:** bad credit

The project studies three fairness families:

1. **Independence**, represented by demographic parity.
2. **Separation**, represented by equalized odds.
3. **Sufficiency**, represented by positive predictive value and calibration.

The project does **not** attempt to identify a universally fairest model.
Instead, it compares operating points under different fairness definitions
and defends one normative choice for the credit-decision context.

## Submission artifacts

The main submission artifacts are:

- Pre-computation commitments: [`00_commitments.md`](00_commitments.md)
- Runnable notebook: [`notebooks/fairness_lab.ipynb`](notebooks/fairness_lab.ipynb)
- Exported notebook HTML: [`outputs/fairness_lab.html`](outputs/fairness_lab.html)
- Output tables: [`outputs/tables/`](outputs/tables/)
- Output figures: [`outputs/figures/`](outputs/figures/)
- Final report Markdown: [`report/fairness_report.md`](report/fairness_report.md)
- Final report PDF: [`report/fairness_report.pdf`](report/fairness_report.pdf)

GitHub links:

- Repository: https://github.com/davisjoseph6/fairness-lab.git
- Commitments: https://github.com/davisjoseph6/fairness-lab/blob/main/00_commitments.md
- Notebook: https://github.com/davisjoseph6/fairness-lab/blob/main/notebooks/fairness_lab.ipynb
- Notebook HTML: https://github.com/davisjoseph6/fairness-lab/blob/main/outputs/fairness_lab.html
- Tables: https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/tables
- Figures: https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/figures
- Report PDF: https://github.com/davisjoseph6/fairness-lab/blob/main/report/fairness_report.pdf

## Group responsibilities

### Davis Joseph

- Dataset loading and inspection
- Preprocessing
- Baseline model
- Mission 1: disparity audit
- Mission 2: impossibility identity

### Malone Mfono

- Mission 3: mitigation methods
- Fairness–accuracy plots
- Mission 5: regulation and deployment limits
- Report formatting and references

### Davis Joseph and Malone Mfono together

- Pre-computation commitments
- Mission 4: operating-point decision
- Interpretation of human costs
- Final report review
- Reproducibility test

## Project structure

```text
fairness-lab/
├── 00_commitments.md
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── notebooks/
│   └── fairness_lab.ipynb
├── src/
│   ├── __init__.py
│   └── fairness_lab.py
├── outputs/
│   ├── environment.json
│   ├── fairness_lab.html
│   ├── tables/
│   ├── figures/
│   └── models/
└── report/
    ├── fairness_report.md
    └── fairness_report.pdf
