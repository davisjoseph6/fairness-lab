# Fairness in Machine Learning Lab

Group members: Da and Ma

## Objective

This project audits and mitigates fairness disparities in a binary
credit-risk classifier trained on the German Credit dataset from OpenML.

The sensitive attribute is age:

- Young: age < 25
- Old: age >= 25

The prediction target is:

- 1: good credit
- 0: bad credit

The project studies three fairness families:

1. Independence, represented by demographic parity
2. Separation, represented by equalized odds
3. Sufficiency, represented by predictive parity or calibration

The project does not attempt to identify a universally fairest model.
It compares operating points under different fairness definitions and
defends one normative choice for the credit-decision context.

## Group responsibilities

### Da

- Dataset loading and inspection
- Preprocessing
- Baseline model
- Mission 1: disparity audit
- Mission 2: impossibility identity

### Ma

- Mission 3: mitigation methods
- Fairness–accuracy plots
- Mission 5: regulation and deployment limits
- Report formatting and references

### Da and Ma together

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
│   └── fairness_lab.py
├── outputs/
│   ├── tables/
│   ├── figures/
│   └── models/
└── report/
    └── fairness_report.md
