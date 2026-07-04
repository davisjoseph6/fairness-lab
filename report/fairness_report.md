## 1. Baseline disparity

Our pre-computation expectation was that applicants below 25 would be
disadvantaged and that deleting age would not eliminate the disparity.
The experiment supported that expectation.

In the baseline model with age included, the old group had a higher
base rate of good credit than the young group: 0.727 compared with
0.553. The model also selected old applicants more often: the old
selection rate was 0.771, while the young selection rate was 0.660,
giving a demographic-parity difference of 0.111.

The disadvantage was not limited to selection rate. The young group had
a lower true-positive rate, 0.731 compared with 0.870 for the old group,
meaning that creditworthy young applicants were less likely to be
approved. The young group also had a higher false-positive rate, 0.571
compared with 0.507, and a lower positive predictive value, 0.613
compared with 0.821. Therefore, the answer to "who is disadvantaged"
depends partly on the metric, but the young group is worse off on the
main opportunity and prediction-quality measures in this run.

Removing age from the predictive features did not solve the problem.
The demographic-parity difference remained exactly 0.111, and the
equalized-odds difference slightly worsened from 0.139 to 0.144. This
rejects the claim that a model cannot discriminate if it does not see
the protected attribute. In this dataset, other variables such as
employment, housing, credit duration, credit history, savings, and
foreign-worker status may plausibly preserve age-related information as
proxies. Removing age mostly removes the explicit variable; it does not
remove all age-related structure from the data.

## 2. Calibration and equal-error impossibility

Using the baseline model, we verified Chouldechova's identity separately
for the old and young groups. For the old group, the observed false
positive rate was 0.507246 and the right-hand side of the identity was
also 0.507246, with only floating-point error. For the young group, the
observed false positive rate was 0.571429 and the identity also returned
0.571429.

The base rates were not equal. The old group had a positive-class base
rate of 0.727273, while the young group had a base rate of 0.553191.
This difference matters because the identity links the false positive
rate to the base rate, the positive predictive value, and the false
negative rate.

To demonstrate the conflict numerically, we used the pooled baseline
PPV, 0.792035, as a common PPV value for both groups. Under that common
PPV, while holding the observed false negative rates fixed, the identity
forced the old group's FPR to 0.608858 and the young group's FPR to
0.237563. The resulting FPR gap was 0.371294.

Therefore, the conflict is not merely a bug in our classifier. It is a
counting relationship. When groups have different base rates, equal
calibration or equal PPV does not generally coexist with equal error
rates. A stakeholder demand for both equal calibration and equal FPR is
therefore mathematically incompatible in ordinary non-perfect settings.
