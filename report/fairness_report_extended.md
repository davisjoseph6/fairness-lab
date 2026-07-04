# Fairness in Credit-Risk Classification

**Group members:** Davis Joseph and Malone Mfono

**Institution:** Aivancity

**Professor:** Professor David Appadourai

**GitHub repository:** https://github.com/davisjoseph6/fairness-lab.git

## Introduction and methodology

This report audits fairness in a binary credit-risk classifier trained
on the German Credit dataset. The target is encoded as good credit = 1
and bad credit = 0. The sensitive attribute is age, divided into young
applicants, defined as age below 25, and old applicants, defined as age
25 or above.

The experiment studies three families of fairness. Demographic parity
belongs to independence and asks whether groups receive positive
predictions at similar rates. Equalized odds belongs to separation and
asks whether error rates are similar across groups conditional on the
true label. Positive predictive value and calibration belong to
sufficiency and ask whether a positive prediction means the same thing
across groups.

The notebook uses one fixed train/test split, one random seed, one
positive-class definition, and one age-group definition. Age is retained
separately for auditing even when it is removed from the predictive
features. This distinction is central to the lab: deleting a sensitive
column from the model is not the same thing as proving that the model no
longer produces group disparities.

## Repository and submission artifacts

The full project repository and submission artifacts are available here:

- Repository: https://github.com/davisjoseph6/fairness-lab.git
- Pre-computation commitments: https://github.com/davisjoseph6/fairness-lab/blob/main/00_commitments.md
- Runnable notebook: https://github.com/davisjoseph6/fairness-lab/blob/main/notebooks/fairness_lab.ipynb
- Exported notebook HTML: https://github.com/davisjoseph6/fairness-lab/blob/main/outputs/fairness_lab.html
- Output tables: https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/tables
- Output figures: https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/figures
- Final report PDF: https://github.com/davisjoseph6/fairness-lab/blob/main/report/fairness_report.pdf


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

## 3. Mitigation and fairness–accuracy frontiers

The mitigation results show that "make it fair" is not a complete
technical instruction. Each method changed a different fairness family.

The baseline model without age had accuracy 0.746667, demographic-parity
difference 0.111177, and equalized-odds difference 0.144231. Reweighing
kept the same accuracy, 0.746667, while reducing demographic-parity
difference to 0.076528 and equalized-odds difference to 0.111801. This
was the best equalized-odds result among the main methods.

Among the main named mitigation methods, ExponentiatedGradient with a
demographic-parity constraint produced the highest accuracy, 0.766667,
and almost eliminated the demographic-parity difference, reducing it to
0.000673. In the epsilon sweep, the related EG-DP eps=0.2 point reached
accuracy 0.770000 with demographic-parity difference 0.003280, but its
equalized-odds difference remained 0.173913. This reinforces the same
trade-off: the strongest demographic-parity points did not also minimise
equalized odds.

For the main ExponentiatedGradient demographic-parity model, the
improvement in selection parity came with a larger false-positive-rate
gap: the old group's FPR was 0.492754, while the young group's FPR was
0.666667. This shows that improving independence did not automatically
improve separation.

ExponentiatedGradient with an equalized-odds constraint did not perform
best on equalized odds on the test set. It achieved accuracy 0.750000
and demographic-parity difference 0.003280, but its equalized-odds
difference was 0.192547. The epsilon sweep also showed that the EG-EO
points kept equalized-odds difference around 0.192547 on this test set.
This reminds us that a constraint fitted on the training data does not
guarantee the smallest test-set fairness gap.

ThresholdOptimizer with demographic parity achieved accuracy 0.740000,
demographic-parity difference 0.074847, and equalized-odds difference
0.283644. ThresholdOptimizer with equalized odds achieved accuracy
0.733333, demographic-parity difference 0.005214, and equalized-odds
difference 0.157350. Both threshold-optimisation methods use
group-specific decision rules, so they must be interpreted differently
from a single common classifier.

The two frontier plots also show that demographic parity and equalized
odds are not one single fairness axis.

![Accuracy versus demographic-parity difference](../outputs/figures/mission3_dp_accuracy_frontier.png)

![Accuracy versus equalized-odds difference](../outputs/figures/mission3_eo_accuracy_frontier.png)

No method eliminated both fairness gaps without cost. The smallest
demographic-parity difference among the main methods came from
ExponentiatedGradient with DemographicParity, but its equalized-odds
difference worsened relative to the baseline without age. The smallest
equalized-odds difference among the main methods came from Reweighing,
but it did not eliminate demographic-parity difference. The two frontier
plots therefore represent different trade-offs rather than one
universal fairness frontier.

## 4. Chosen operating point

We choose the Reweighing operating point. We do not claim that it is the
fairest model in general. We choose it because it best matches our
normative priority: reducing unequal error rates and reducing the
false-rejection burden on creditworthy young applicants, while avoiding
a fully automated deployment decision.

Compared with the baseline without age, Reweighing kept accuracy
unchanged at 0.746667. It reduced demographic-parity difference from
0.111177 to 0.076528 and reduced equalized-odds difference from
0.144231 to 0.111801, the smallest equalized-odds difference among the
main methods. It also improved the young group's true-positive rate
from 0.730769 to 0.769231, meaning that more creditworthy young
applicants were approved.

The cost is not zero. The young group's false-positive rate increased
from 0.571429 to 0.619048, and its positive predictive value decreased
slightly from 0.612903 to 0.606061. In human terms, this means that the
system may approve more young applicants who later turn out to be bad
credit risks. We assign more of that residual cost to the bank through
manual review, monitoring, and limited additional credit risk, rather
than assigning it entirely to creditworthy young applicants who would
otherwise be wrongly rejected.

We also reject a purely metric-based deployment decision. The label
"good credit" is itself historical. It reflects past lending and
repayment patterns, not a neutral moral truth. Applicants who were
previously denied credit may be missing from the outcome data, and past
credit access may have shaped the target itself. Therefore, this model
should not be treated as a self-justifying decision-maker. At most, it
could be used as a decision-support tool with human oversight, reason
codes, and an appeal route.

## 5. Regulation and deployment limits

We would not deploy the chosen model as a fully automated credit
decision system. If used at all, it should be used only as a
decision-support tool with documentation, human review, bias monitoring,
reason codes, and an appeal route.

Under the EU AI Act, an AI system used to evaluate the creditworthiness
of natural persons or establish their credit score is a high-risk AI
system, except where the system is used for detecting financial fraud.
Our system is therefore high-risk if used for real creditworthiness
assessment.

Two concrete obligations follow. First, the system requires data and
bias governance. The provider would need to document the origin,
preparation, assumptions, suitability, and representativeness of the
training, validation, and test data. It would also need to examine
possible biases and take appropriate measures to detect, prevent, and
mitigate them. Our Mission 1 result shows why deleting age is not an
adequate governance response: the demographic-parity difference remained
0.111177 after age was removed from the predictive features.

Second, the system requires human oversight. We would implement a manual
review band and appeal route. All rejected applications, and all
borderline cases within a predefined score interval, should be reviewed
by a trained credit officer. The officer should receive the model score,
reason codes, the applicant file, known model limitations, and a warning
against automation bias. The reviewer must be able to override, reverse,
or disregard the model output.

GDPR also limits deployment. A fully automated credit refusal may fall
under the restriction on solely automated decisions that produce legal
or similarly significant effects. Therefore, the system should not issue
final automatic rejections. Rejected applicants should receive
meaningful information, a reason code, the ability to express their
point of view, human intervention, and a route to contest the decision.

Finally, age is not a GDPR Article 9 special-category attribute. Article
9 lists categories such as racial or ethnic origin, political opinions,
religious or philosophical beliefs, trade-union membership, genetic
data, biometric identification data, health data, sex life, and sexual
orientation. However, age is still relevant for discrimination analysis.
The legally and ethically defensible approach is not to delete age
blindly, but to retain it in a controlled way for auditing and bias
monitoring, with access controls and documentation. This reconciles the
legal analysis with Mission 1: removing age from the model did not
remove age-related disparity.

## Conclusion

This lab shows why "make the model fair" is not a complete instruction.
The baseline model disadvantaged young applicants under several metrics:
young applicants had a lower base rate, lower selection rate, lower
TPR, higher FPR, and lower PPV than old applicants. Removing age from
the predictive features did not remove the disparity.

The impossibility result also appeared numerically. Chouldechova's
identity reproduced the observed FPR in each group, and forcing a common
PPV produced a large forced FPR gap because the groups had different
base rates.

The mitigation results confirmed that fairness gains are transfers, not
free lunches. ExponentiatedGradient with DemographicParity almost
eliminated the demographic-parity gap, but worsened the equalized-odds
gap. Reweighing gave the best equalized-odds result among the main
methods without reducing accuracy, so we selected it as the most
defensible operating point for our stated values.

Our deployment conclusion is cautious. The model should not be deployed
as a fully automated credit-decision system. At most, it could be used
as a documented decision-support tool with bias monitoring, human
review, reason codes, and an appeal route.

\newpage

## Technical appendices

The main report contains the defended fairness argument. These
appendices provide the supporting audit trail: commitments, generated
outputs, environment information, regulatory mapping, code excerpts,
and reproduction instructions.

The appendices are not meant to replace the argument. They make the
numerical and technical claims verifiable.



\newpage

## Appendix A — Submission metadata and artifact links

**Group members:** Davis Joseph and Malone Mfono

**Institution:** Aivancity

**Professor:** Professor David Appadourai

**Repository:** https://github.com/davisjoseph6/fairness-lab.git

**Required artifacts:**

- Pre-computation commitments: https://github.com/davisjoseph6/fairness-lab/blob/main/00_commitments.md
- Runnable notebook: https://github.com/davisjoseph6/fairness-lab/blob/main/notebooks/fairness_lab.ipynb
- Exported notebook HTML: https://github.com/davisjoseph6/fairness-lab/blob/main/outputs/fairness_lab.html
- Output tables: https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/tables
- Output figures: https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/figures
- Final report PDF: https://github.com/davisjoseph6/fairness-lab/blob/main/report/fairness_report.pdf



\newpage

## Appendix B — Pre-computation commitments

The full pre-computation commitment file is reproduced below.

```markdown

```



\newpage

## Appendix C — Reproducibility environment

Environment metadata generated by the notebook:

```json
{
  "python_version": "3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]",
  "python_implementation": "CPython",
  "operating_system": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39",
  "machine": "x86_64",
  "random_state": 42,
  "test_size": 0.3,
  "positive_label": 1,
  "negative_label": 0,
  "age_threshold": 25,
  "young_group_definition": "age < 25",
  "old_group_definition": "age >= 25",
  "package_versions": {
    "numpy": "2.5.0",
    "pandas": "3.0.3",
    "matplotlib": "3.11.0",
    "scikit-learn": "1.9.0",
    "fairlearn": "0.14.0",
    "openml": "0.15.1",
    "jupyterlab": "4.6.0",
    "ipykernel": "7.3.0"
  }
}
```

Direct dependency list:

```text
numpy
pandas
matplotlib
scikit-learn
fairlearn
openml
jupyterlab
ipykernel
```

Exact lock file used for the run:

```text
anyio==4.14.1
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
captum==0.9.0
certifi==2026.6.17
cffi==2.0.0
charset-normalizer==3.4.7
cloudpickle==3.1.2
comm==0.2.3
contourpy==1.3.3
cuda-bindings==13.3.1
cuda-pathfinder==1.5.5
cuda-toolkit==13.0.2
cycler==0.12.1
debugpy==1.8.21
decorator==5.3.1
defusedxml==0.7.1
executing==2.2.1
fairlearn==0.14.0
fastjsonschema==2.21.2
filelock==3.29.4
fonttools==4.63.0
fqdn==1.5.1
fsspec==2026.6.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.18
ImageIO==2.37.3
ipykernel==7.3.0
ipython==9.15.0
ipython_pygments_lexers==1.1.1
isoduration==20.11.0
jedi==0.20.0
Jinja2==3.1.6
joblib==1.5.3
json5==0.15.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter_builder==1.0.2
jupyter_client==8.9.1
jupyter_core==5.9.1
jupyter_server==2.20.0
jupyter_server_terminals==0.5.4
jupyterlab==4.6.1
jupyterlab_pygments==0.3.0
jupyterlab_server==2.28.0
kiwisolver==1.5.0
lark==1.3.1
lazy-loader==0.5
liac-arff==2.5.0
lime==0.2.0.1
llvmlite==0.47.0
MarkupSafe==3.0.3
matplotlib==3.11.0
matplotlib-inline==0.2.2
minio==7.2.20
mistune==3.3.2
mpmath==1.3.0
narwhals==2.22.1
nbclient==0.11.0
nbconvert==7.17.1
nbformat==5.10.4
nest-asyncio2==1.7.2
networkx==3.6.1
notebook_shim==0.2.4
numba==0.65.1
numpy==2.4.6
nvidia-cublas==13.1.1.3
nvidia-cuda-cupti==13.0.85
nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96
nvidia-cudnn-cu13==9.20.0.48
nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6
nvidia-curand==10.4.0.35
nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3
nvidia-cusparselt-cu13==0.8.1
nvidia-nccl-cu12==2.30.7
nvidia-nccl-cu13==2.29.7
nvidia-nvjitlink==13.0.88
nvidia-nvshmem-cu13==3.4.5
nvidia-nvtx==13.0.85
openml==0.15.1
packaging==26.2
pandas==3.0.3
pandocfilters==1.5.1
parso==0.8.7
pexpect==4.9.0
pillow==12.2.0
platformdirs==4.10.0
prometheus_client==0.25.0
prompt_toolkit==3.0.52
psutil==7.2.2
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==24.0.0
pycparser==3.0
pycryptodome==3.23.0
Pygments==2.20.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
python-json-logger==4.1.0
PyYAML==6.0.3
pyzmq==27.1.0
referencing==0.37.0
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.6.3
scikit-image==0.26.0
scikit-learn==1.9.0
scipy==1.18.0
Send2Trash==2.1.0
setuptools==81.0.0
shap==0.52.0
six==1.17.0
slicer==0.0.8
soupsieve==2.8.4
stack-data==0.6.3
sympy==1.14.0
tabulate==0.10.0
terminado==0.18.1
threadpoolctl==3.6.0
tifffile==2026.6.1
tinycss2==1.5.1
torch==2.12.1
torchvision==0.27.1
tornado==6.5.7
tqdm==4.68.3
traitlets==5.15.1
triton==3.7.1
typing_extensions==4.15.0
tzdata==2026.2
ucimlrepo==0.0.7
uri-template==1.3.0
urllib3==2.7.0
wcwidth==0.8.2
webcolors==25.10.0
webencodings==0.5.1
websocket-client==1.9.0
wheel==0.47.0
xgboost==3.3.0
xmltodict==1.0.4
```



\newpage

## Appendix D — Project file manifest

The following files form the reproducible submission package.

- `00_commitments.md`
- `README.md`
- `notebooks/fairness_lab.ipynb`
- `outputs/environment.json`
- `outputs/fairness_lab.html`
- `outputs/figures/.gitkeep`
- `outputs/figures/mission3_dp_accuracy_frontier.png`
- `outputs/figures/mission3_eo_accuracy_frontier.png`
- `outputs/tables/.gitkeep`
- `outputs/tables/mission1_age_removed_group_metrics.csv`
- `outputs/tables/mission1_age_removed_summary.csv`
- `outputs/tables/mission1_baseline_group_metrics.csv`
- `outputs/tables/mission1_baseline_summary.csv`
- `outputs/tables/mission1_model_comparison.csv`
- `outputs/tables/mission1_sensitive_group_counts.csv`
- `outputs/tables/mission1_train_test_indices.csv`
- `outputs/tables/mission2_chouldechova_identity.csv`
- `outputs/tables/mission2_equal_ppv_counterfactual.csv`
- `outputs/tables/mission2_equal_ppv_equal_fnr_counterfactual.csv`
- `outputs/tables/mission3_eg_epsilon_sweep.csv`
- `outputs/tables/mission3_frontier_points.csv`
- `outputs/tables/mission3_model_comparison.csv`
- `outputs/tables/mission3_reweighing_weights.csv`
- `outputs/tables/mission4_chosen_delta.csv`
- `outputs/tables/mission4_chosen_operating_point.csv`
- `outputs/tables/mission4_chosen_summary.json`
- `outputs/tables/mission5_oversight_measure.json`
- `outputs/tables/mission5_regulatory_obligations.csv`
- `report/fairness_report.md`
- `report/fairness_report.pdf`
- `requirements-lock.txt`
- `requirements.txt`
- `scripts/build_extended_report.py`
- `src/fairness_lab.py`



\newpage

## Appendix E — Mission 1: group counts and split

These tables document the sensitive-group counts and the fixed train/test
split. The same split is reused across comparable models.

### Sensitive group counts

```csv
age_group,count
old,851
young,149
```

### Train/test indices excerpt

```csv
index,split
328,train
891,train
255,train
243,train
492,train
597,train
7,train
951,train
224,train
317,train
127,train
587,train
967,train
355,train
896,train
841,train
281,train
702,train
402,train
293,train
96,train
20,train
855,train
104,train
833,train
726,train
197,train
793,train
458,train
625,train
8,train
429,train
463,train
494,train
772,train
745,train
382,train
663,train
796,train
158,train
519,train
346,train
758,train
120,train
516,train
455,train
750,train
323,train
76,train
579,train
925,train
392,train
865,train
497,train
874,train
965,train
16,train
309,train
684,train
984,train
904,train
246,train
948,train
832,train
799,train
251,train
722,train
571,train
318,train
854,train
765,train
644,train
527,train
662,train
64,train
698,train
867,train
664,train
515,train
329,train
103,train
291,train
542,train
167,train
403,train
704,train
754,train
568,train
21,train
737,train
840,train
49,train
864,train
351,train
915,train
863,train
933,train
672,train
426,train
512,train
498,train
380,train
414,train
229,train
307,train
504,train
880,train
553,train
132,train
185,train
817,train
215,train
408,train
205,train
882,train
210,train
616,train
247,train
781,train
400,train
45,train
32,train
910,train
791,train
14,train
52,train
539,train
666,train
771,train
... 741 lines omitted; full file is in the repository ...
892,test
194,test
907,test
768,test
931,test
59,test
802,test
738,test
169,test
10,test
302,test
468,test
214,test
509,test
636,test
743,test
834,test
235,test
376,test
461,test
543,test
624,test
33,test
245,test
989,test
430,test
162,test
297,test
615,test
131,test
231,test
947,test
714,test
968,test
806,test
252,test
261,test
871,test
853,test
875,test
391,test
219,test
690,test
674,test
850,test
518,test
620,test
477,test
341,test
125,test
883,test
211,test
727,test
687,test
777,test
659,test
952,test
464,test
580,test
523,test
633,test
838,test
141,test
955,test
798,test
721,test
4,test
333,test
653,test
876,test
812,test
782,test
365,test
577,test
348,test
752,test
554,test
321,test
962,test
514,test
718,test
475,test
651,test
151,test
394,test
257,test
922,test
548,test
619,test
897,test
29,test
809,test
77,test
262,test
180,test
544,test
657,test
998,test
868,test
442,test
164,test
378,test
635,test
561,test
25,test
556,test
296,test
69,test
119,test
377,test
395,test
368,test
630,test
260,test
200,test
370,test
549,test
540,test
67,test
263,test
418,test
963,test
47,test
870,test
356,test
19,test
208,test
627,test
357,test
964,test
```



\newpage

## Appendix F — Mission 1: baseline and age-removal audit

These tables support the Mission 1 claim that young applicants were
disadvantaged under several metrics and that removing age did not remove
the disparity.

### Baseline group metrics

```csv
model,group,n,base_rate,selection_rate,TPR,FPR,FNR,PPV,TN,FP,FN,TP
Baseline with age,old,253.0,0.7272727272727273,0.7707509881422925,0.8695652173913043,0.5072463768115942,0.13043478260869565,0.8205128205128205,34,35,24,160
Baseline with age,young,47.0,0.5531914893617021,0.6595744680851063,0.7307692307692307,0.5714285714285714,0.2692307692307692,0.6129032258064516,9,12,7,19
```

### Baseline summary

```csv
model,accuracy,selection_rate,DP_difference,EO_difference
Baseline with age,0.74,0.7533333333333333,0.11117652005718615,0.1387959866220736
```

### Age-removed group metrics

```csv
model,group,n,base_rate,selection_rate,TPR,FPR,FNR,PPV,TN,FP,FN,TP
Model without age,old,253.0,0.7272727272727273,0.7707509881422925,0.875,0.4927536231884058,0.125,0.8256410256410256,35,34,23,161
Model without age,young,47.0,0.5531914893617021,0.6595744680851063,0.7307692307692307,0.5714285714285714,0.2692307692307692,0.6129032258064516,9,12,7,19
```

### Age-removed summary

```csv
model,accuracy,selection_rate,DP_difference,EO_difference
Model without age,0.7466666666666667,0.7533333333333333,0.11117652005718615,0.14423076923076927
```

### Mission 1 model comparison

```csv
model,accuracy,selection_rate,DP_difference,EO_difference
Baseline with age,0.74,0.7533333333333333,0.11117652005718615,0.1387959866220736
Model without age,0.7466666666666667,0.7533333333333333,0.11117652005718615,0.14423076923076927
```



\newpage

## Appendix G — Mission 1 interpretation matrix

The baseline model selected old applicants more often and produced
better opportunity and prediction-quality metrics for them. In the
baseline model, the young group had lower base rate, lower selection
rate, lower TPR, higher FPR, and lower PPV.

The age-removal experiment is central. Age was removed from the
predictive features, but age was retained as the sensitive attribute for
auditing. The demographic-parity difference did not disappear. The
equalized-odds difference also did not disappear. This supports the
claim that fairness through unawareness is not a sufficient defence.

Plausible proxy features include employment, housing, credit duration,
credit history, savings status, and foreign-worker status. The report
does not claim that each proxy causally explains the disparity; it only
states that the model can preserve age-related information through
correlated variables.



\newpage

## Appendix H — Mission 2: Chouldechova identity outputs

This appendix records the numerical verification of the identity:

FPR = (p / (1 - p)) × ((1 - PPV) / PPV) × (1 - FNR)

The identity reproduces the observed FPR in both groups.

### Identity table

```csv
group,TN,FP,FN,TP,p_base_rate,PPV,FNR,observed_FPR,identity_RHS,absolute_error
old,34.0,35.0,24.0,160.0,0.7272727272727273,0.8205128205128205,0.13043478260869565,0.5072463768115942,0.5072463768115945,2.220446049250313e-16
young,9.0,12.0,7.0,19.0,0.5531914893617021,0.6129032258064516,0.2692307692307692,0.5714285714285714,0.5714285714285714,0.0
```

### Equal-PPV counterfactual

```csv
group,base_rate_p,observed_PPV,common_PPV_used,observed_FNR,observed_FPR,forced_FPR_under_common_PPV
old,0.7272727272727273,0.8205128205128205,0.7920353982300885,0.13043478260869565,0.5072463768115942,0.6088575823819934
young,0.5531914893617021,0.6129032258064516,0.7920353982300885,0.2692307692307692,0.5714285714285714,0.23756318169725996
```



\newpage

## Appendix I — Mission 2 interpretation

The old group had a higher base rate than the young group. When the
groups have different base rates, equal PPV and equal FPR generally
cannot both be imposed in ordinary non-perfect conditions.

The counterfactual used the pooled PPV as a common PPV value. Under that
common PPV, the identity forced the FPR values far apart. This is the
numerical form of the impossibility result used in the report.

The key point is that the conflict is not a programming error. It is a
counting relationship involving prevalence, predictive value, false
negative rate, and false positive rate.



\newpage

## Appendix J — Mission 3: reweighing and mitigation setup

Mission 3 used age-removed predictive features. The age group remained
available separately for auditing and for fairness constraints.

Reweighing changes the effective training distribution by giving
different weights to group-label combinations. ExponentiatedGradient
imposes an in-processing fairness constraint. ThresholdOptimizer changes
post-processing decision thresholds and may use group-specific decision
rules.

### Reweighing weights

```csv
group,label,group_count,label_count,joint_count,weight
old,0,598,210,170,1.0552941176470587
old,1,598,490,428,0.97803738317757
young,0,102,210,40,0.765
young,1,102,490,62,1.1516129032258065
```



\newpage

## Appendix K — Mission 3: main mitigation comparison

The following comparison table is the main quantitative basis for the
Mission 3 discussion.

```csv
method,fairness_family_targeted,accuracy,selection_rate,DP_difference,EO_difference,old_TPR,young_TPR,TPR_gap_abs,old_FPR,young_FPR,FPR_gap_abs,old_PPV,young_PPV,notes
Baseline without age,None,0.7466666666666667,0.7533333333333333,0.11117652005718615,0.14423076923076927,0.875,0.7307692307692307,0.14423076923076927,0.4927536231884058,0.5714285714285714,0.07867494824016558,0.8256410256410256,0.6129032258064516,Reference model for mitigation; age removed from predictive features.
Reweighing,Pre-processing / distribution balancing,0.7466666666666667,0.7666666666666667,0.07652846690774529,0.11180124223602483,0.8804347826086957,0.7692307692307693,0.1112040133779264,0.5072463768115942,0.6190476190476191,0.11180124223602483,0.8223350253807107,0.6060606060606061,Changes sample weights in training.
ExponentiatedGradient — DemographicParity,Independence,0.7666666666666667,0.7866666666666666,0.000672777731057117,0.1739130434782608,0.8967391304347826,0.8846153846153846,0.012123745819398013,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.8291457286432161,0.6216216216216216,In-processing constraint targeting demographic parity.
ExponentiatedGradient — EqualizedOdds,Separation,0.75,0.79,0.0032797914389033345,0.1925465838509317,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.82,0.5945945945945946,In-processing constraint targeting equalized odds.
ThresholdOptimizer — DemographicParity,Independence,0.74,0.7666666666666667,0.0748465225801026,0.28364389233954446,0.8586956521739131,0.8846153846153846,0.0259197324414715,0.4782608695652174,0.7619047619047619,0.28364389233954446,0.8272251308900523,0.5897435897435898,Post-processing with group-specific decision rules.
ThresholdOptimizer — EqualizedOdds,Separation,0.7333333333333333,0.8466666666666667,0.005214027415692546,0.15734989648033126,0.9184782608695652,0.8846153846153846,0.03386287625418061,0.6521739130434783,0.8095238095238095,0.15734989648033126,0.7897196261682243,0.575,Post-processing with group-specific decision rules.
```



\newpage

## Appendix L — Mission 3: epsilon sweep and frontier data

The epsilon sweep provides additional operating points for the
fairness-accuracy frontier.

### ExponentiatedGradient epsilon sweep

```csv
method,fairness_family_targeted,accuracy,selection_rate,DP_difference,EO_difference,old_TPR,young_TPR,TPR_gap_abs,old_FPR,young_FPR,FPR_gap_abs,old_PPV,young_PPV,notes
EG-DP eps=0.01,Independence,0.7533333333333333,0.7866666666666666,0.024556387183584216,0.14492753623188404,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.6666666666666666,0.14492753623188404,0.82,0.6111111111111112,Epsilon sweep point for frontier plot.
EG-DP eps=0.02,Independence,0.7666666666666667,0.7866666666666666,0.000672777731057117,0.1739130434782608,0.8967391304347826,0.8846153846153846,0.012123745819398013,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.8291457286432161,0.6216216216216216,Epsilon sweep point for frontier plot.
EG-DP eps=0.05,Independence,0.7533333333333333,0.7866666666666666,0.024556387183584216,0.14492753623188404,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.6666666666666666,0.14492753623188404,0.82,0.6111111111111112,Epsilon sweep point for frontier plot.
EG-DP eps=0.1,Independence,0.7666666666666667,0.7866666666666666,0.000672777731057117,0.1739130434782608,0.8967391304347826,0.8846153846153846,0.012123745819398013,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.8291457286432161,0.6216216216216216,Epsilon sweep point for frontier plot.
EG-DP eps=0.2,Independence,0.77,0.79,0.0032797914389033345,0.1739130434782608,0.9021739130434783,0.8846153846153846,0.01755852842809369,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.83,0.6216216216216216,Epsilon sweep point for frontier plot.
EG-EO eps=0.01,Separation,0.75,0.79,0.028508956353544668,0.1925465838509317,0.8967391304347826,0.8076923076923077,0.08904682274247488,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5833333333333334,Epsilon sweep point for frontier plot.
EG-EO eps=0.02,Separation,0.75,0.79,0.0032797914389033345,0.1925465838509317,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.82,0.5945945945945946,Epsilon sweep point for frontier plot.
EG-EO eps=0.05,Separation,0.75,0.79,0.028508956353544668,0.1925465838509317,0.8967391304347826,0.8076923076923077,0.08904682274247488,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5833333333333334,Epsilon sweep point for frontier plot.
EG-EO eps=0.1,Separation,0.75,0.79,0.028508956353544668,0.1925465838509317,0.8967391304347826,0.8076923076923077,0.08904682274247488,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5833333333333334,Epsilon sweep point for frontier plot.
EG-EO eps=0.2,Separation,0.7533333333333333,0.7933333333333333,0.007232360608863786,0.1925465838509317,0.8967391304347826,0.8461538461538461,0.05058528428093645,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5945945945945946,Epsilon sweep point for frontier plot.
```

### Frontier points

```csv
method,fairness_family_targeted,accuracy,selection_rate,DP_difference,EO_difference,old_TPR,young_TPR,TPR_gap_abs,old_FPR,young_FPR,FPR_gap_abs,old_PPV,young_PPV,notes,plot_group
Baseline without age,None,0.7466666666666667,0.7533333333333333,0.11117652005718615,0.14423076923076927,0.875,0.7307692307692307,0.14423076923076927,0.4927536231884058,0.5714285714285714,0.07867494824016558,0.8256410256410256,0.6129032258064516,Reference model for mitigation; age removed from predictive features.,main_methods
Reweighing,Pre-processing / distribution balancing,0.7466666666666667,0.7666666666666667,0.07652846690774529,0.11180124223602483,0.8804347826086957,0.7692307692307693,0.1112040133779264,0.5072463768115942,0.6190476190476191,0.11180124223602483,0.8223350253807107,0.6060606060606061,Changes sample weights in training.,main_methods
ExponentiatedGradient — DemographicParity,Independence,0.7666666666666667,0.7866666666666666,0.000672777731057117,0.1739130434782608,0.8967391304347826,0.8846153846153846,0.012123745819398013,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.8291457286432161,0.6216216216216216,In-processing constraint targeting demographic parity.,main_methods
ExponentiatedGradient — EqualizedOdds,Separation,0.75,0.79,0.0032797914389033345,0.1925465838509317,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.82,0.5945945945945946,In-processing constraint targeting equalized odds.,main_methods
ThresholdOptimizer — DemographicParity,Independence,0.74,0.7666666666666667,0.0748465225801026,0.28364389233954446,0.8586956521739131,0.8846153846153846,0.0259197324414715,0.4782608695652174,0.7619047619047619,0.28364389233954446,0.8272251308900523,0.5897435897435898,Post-processing with group-specific decision rules.,main_methods
ThresholdOptimizer — EqualizedOdds,Separation,0.7333333333333333,0.8466666666666667,0.005214027415692546,0.15734989648033126,0.9184782608695652,0.8846153846153846,0.03386287625418061,0.6521739130434783,0.8095238095238095,0.15734989648033126,0.7897196261682243,0.575,Post-processing with group-specific decision rules.,main_methods
EG-DP eps=0.01,Independence,0.7533333333333333,0.7866666666666666,0.024556387183584216,0.14492753623188404,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.6666666666666666,0.14492753623188404,0.82,0.6111111111111112,Epsilon sweep point for frontier plot.,eg_sweep
EG-DP eps=0.02,Independence,0.7666666666666667,0.7866666666666666,0.000672777731057117,0.1739130434782608,0.8967391304347826,0.8846153846153846,0.012123745819398013,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.8291457286432161,0.6216216216216216,Epsilon sweep point for frontier plot.,eg_sweep
EG-DP eps=0.05,Independence,0.7533333333333333,0.7866666666666666,0.024556387183584216,0.14492753623188404,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.6666666666666666,0.14492753623188404,0.82,0.6111111111111112,Epsilon sweep point for frontier plot.,eg_sweep
EG-DP eps=0.1,Independence,0.7666666666666667,0.7866666666666666,0.000672777731057117,0.1739130434782608,0.8967391304347826,0.8846153846153846,0.012123745819398013,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.8291457286432161,0.6216216216216216,Epsilon sweep point for frontier plot.,eg_sweep
EG-DP eps=0.2,Independence,0.77,0.79,0.0032797914389033345,0.1739130434782608,0.9021739130434783,0.8846153846153846,0.01755852842809369,0.4927536231884058,0.6666666666666666,0.1739130434782608,0.83,0.6216216216216216,Epsilon sweep point for frontier plot.,eg_sweep
EG-EO eps=0.01,Separation,0.75,0.79,0.028508956353544668,0.1925465838509317,0.8967391304347826,0.8076923076923077,0.08904682274247488,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5833333333333334,Epsilon sweep point for frontier plot.,eg_sweep
EG-EO eps=0.02,Separation,0.75,0.79,0.0032797914389033345,0.1925465838509317,0.8913043478260869,0.8461538461538461,0.04515050167224077,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.82,0.5945945945945946,Epsilon sweep point for frontier plot.,eg_sweep
EG-EO eps=0.05,Separation,0.75,0.79,0.028508956353544668,0.1925465838509317,0.8967391304347826,0.8076923076923077,0.08904682274247488,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5833333333333334,Epsilon sweep point for frontier plot.,eg_sweep
EG-EO eps=0.1,Separation,0.75,0.79,0.028508956353544668,0.1925465838509317,0.8967391304347826,0.8076923076923077,0.08904682274247488,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5833333333333334,Epsilon sweep point for frontier plot.,eg_sweep
EG-EO eps=0.2,Separation,0.7533333333333333,0.7933333333333333,0.007232360608863786,0.1925465838509317,0.8967391304347826,0.8461538461538461,0.05058528428093645,0.5217391304347826,0.7142857142857143,0.1925465838509317,0.8208955223880597,0.5945945945945946,Epsilon sweep point for frontier plot.,eg_sweep
```



\newpage

## Appendix M — Mission 3 frontier figures

The two figures below are repeated in the appendix for readability. They
should be interpreted as two different frontiers, not as one generic
fairness score.

### Accuracy versus demographic-parity difference

![Accuracy versus demographic-parity difference](../outputs/figures/mission3_dp_accuracy_frontier.png)

\newpage

### Accuracy versus equalized-odds difference

![Accuracy versus equalized-odds difference](../outputs/figures/mission3_eo_accuracy_frontier.png)



\newpage

## Appendix N — Mission 4: chosen operating point

The chosen operating point is Reweighing. The group chose it because it
best matched the stated normative priority: reducing unequal error rates
and false-rejection burden without reducing accuracy.

### Chosen operating point

```csv
method,fairness_family_targeted,accuracy,selection_rate,DP_difference,EO_difference,old_TPR,young_TPR,TPR_gap_abs,old_FPR,young_FPR,FPR_gap_abs,old_PPV,young_PPV,notes
Reweighing,Pre-processing / distribution balancing,0.7466666666666667,0.7666666666666667,0.07652846690774529,0.11180124223602483,0.8804347826086957,0.7692307692307693,0.1112040133779264,0.5072463768115942,0.6190476190476191,0.11180124223602483,0.8223350253807107,0.6060606060606061,Changes sample weights in training.
```

### Delta from baseline without age

```csv
metric,baseline_without_age,chosen_reweighing,change
accuracy,0.7466666666666667,0.7466666666666667,0.0
DP_difference,0.11117652005718615,0.07652846690774529,-0.03464805314944086
EO_difference,0.14423076923076927,0.11180124223602483,-0.03242952699474444
young_TPR,0.7307692307692307,0.7692307692307693,0.03846153846153855
young_FPR,0.5714285714285714,0.6190476190476191,0.04761904761904767
young_PPV,0.6129032258064516,0.6060606060606061,-0.006842619745845546
```

### Chosen summary

```json
{
  "chosen_method": "Reweighing",
  "chosen_criterion": "Separation-oriented fairness: reduce unequal error rates, especially the false-rejection burden on creditworthy young applicants.",
  "main_benefit": "Equalized-odds difference decreased from 0.144231 to 0.111801 without reducing accuracy.",
  "secondary_benefit": "Demographic-parity difference decreased from 0.111177 to 0.076528.",
  "human_benefit": "Young applicants who are actually creditworthy are approved more often; young TPR increased from 0.730769 to 0.769231.",
  "main_cost": "Young FPR increased from 0.571429 to 0.619048 and young PPV decreased from 0.612903 to 0.606061.",
  "who_pays": "The bank bears more residual cost through review, monitoring, and possible additional default risk.",
  "deployment_position": "Decision support only, not fully automated approval or rejection."
}
```



\newpage

## Appendix O — Mission 5: regulatory mapping

This appendix records the legal and deployment mapping used in Mission
5. The report conclusion is that the system should not be deployed as a
fully automated credit-decision system.

### Regulatory obligations

```csv
source,article_or_location,obligation_or_rule,project_implementation
EU AI Act,"Annex III, point 5(b)","AI systems used to evaluate the creditworthiness of natural persons or establish credit scores are high-risk, except systems used to detect financial fraud.",Treat this credit-risk classifier as high-risk if used for real creditworthiness assessment.
EU AI Act,Article 10,"High-risk AI systems trained with data require data governance, including examination of possible biases and measures to detect, prevent, and mitigate them.","Document data origin, assumptions, age-group disparities, proxy concerns, mitigation results, and subgroup performance."
EU AI Act,Article 10(5),Special-category personal data may exceptionally be processed for bias detection and correction when strictly necessary and subject to safeguards.,"Age is not a GDPR Article 9 special category, but the same logic supports controlled retention of sensitive auditing attributes rather than blind deletion."
EU AI Act,Article 14,"High-risk AI systems must be designed so that natural persons can effectively oversee them, understand limitations, monitor operation, avoid automation bias, interpret outputs, and override or reverse outputs where appropriate.","Use the model only as decision support. A trained human reviewer must be able to override, reverse, or disregard the model output."
GDPR,Article 22,"Data subjects have rights against decisions based solely on automated processing that produce legal or similarly significant effects. Safeguards include human intervention, expression of the person's point of view, and contestation.","Do not issue final automatic rejections. Provide human review, reason codes, applicant contestation, and an appeal route."
```

### Oversight measure

```json
{
  "oversight_measure": "Manual review band and appeal route",
  "which_cases_are_reviewed": "All rejected applications, plus borderline approvals or rejections within a predefined score interval.",
  "reviewer": "Trained credit officer",
  "reviewer_powers": "The reviewer may approve, reject, request additional documents, or override or reverse the model recommendation.",
  "information_given_to_reviewer": "Model score, reason codes, applicant file, known model limitations, and a warning against automation bias.",
  "applicant_rights": "Rejected applicants receive a reason code and can request human review or contest the decision.",
  "deployment_conclusion": "Do not deploy as a fully automated credit-decision system. Deploy only, if at all, as a documented decision-support tool."
}
```



\newpage

## Appendix P — Shared Python module

The shared Python module contains the reproducibility configuration and
project utilities.

```python
"""Shared configuration and utility functions for the fairness lab.

This module contains project-wide constants and reproducibility helpers.
Dataset loading, model training, fairness auditing and mitigation functions
will be added during the later missions.
"""

from __future__ import annotations

import json
import platform
import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30

POSITIVE_LABEL: int = 1
NEGATIVE_LABEL: int = 0

YOUNG_GROUP: str = "young"
OLD_GROUP: str = "old"
AGE_THRESHOLD: int = 25


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
MODELS_DIR: Path = OUTPUT_DIR / "models"

REPORT_DIR: Path = PROJECT_ROOT / "report"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


def ensure_project_directories() -> None:
    """Create all generated-output directories if they do not exist."""

    directories = (
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REPORT_DIR,
        NOTEBOOKS_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seeds used by Python and NumPy."""

    random.seed(seed)
    np.random.seed(seed)


def define_age_group(age: float | int) -> str:
    """Convert an applicant's age into the lab's sensitive group."""

    numeric_age = float(age)

    if not np.isfinite(numeric_age):
        raise ValueError("Age must be a finite numeric value.")

    if numeric_age < 0:
        raise ValueError("Age cannot be negative.")

    if numeric_age < AGE_THRESHOLD:
        return YOUNG_GROUP

    return OLD_GROUP


def installed_version(distribution_name: str) -> str:
    """Return the installed version of a Python distribution."""

    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def get_environment_info() -> dict[str, Any]:
    """Collect software and reproducibility information."""

    package_names = (
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "fairlearn",
        "openml",
        "jupyterlab",
        "ipykernel",
    )

    package_versions = {
        package_name: installed_version(package_name)
        for package_name in package_names
    }

    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "age_threshold": AGE_THRESHOLD,
        "young_group_definition": f"age < {AGE_THRESHOLD}",
        "old_group_definition": f"age >= {AGE_THRESHOLD}",
        "package_versions": package_versions,
    }


def save_environment_info(destination: Path | None = None) -> Path:
    """Save environment details as a JSON file."""

    ensure_project_directories()

    output_path = destination or (OUTPUT_DIR / "environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    environment_info = get_environment_info()

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(environment_info, file, indent=2, ensure_ascii=False)

    return output_path


def initialise_project() -> dict[str, Any]:
    """Initialise directories, random seeds and environment metadata."""

    ensure_project_directories()
    set_global_seed()

    environment_path = save_environment_info()
    environment_info = get_environment_info()

    print("Fairness lab project initialised.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Environment information: {environment_path}")

    return environment_info


def main() -> None:
    """Run the project-initialisation checks."""

    environment_info = initialise_project()

    print("\nInstalled package versions:")

    for package_name, version in environment_info["package_versions"].items():
        print(f"- {package_name}: {version}")


if __name__ == "__main__":
    main()
```



\newpage

## Appendix Q — Selected notebook code excerpts

### Notebook code cell 5

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
from src.fairness_lab import (
    AGE_THRESHOLD,
    FIGURES_DIR,
    MODELS_DIR,
    NEGATIVE_LABEL,
    OLD_GROUP,
    OUTPUT_DIR,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TABLES_DIR,
    TEST_SIZE,
    YOUNG_GROUP,
    define_age_group,
    initialise_project,
)
```

\newpage

### Notebook code cell 8

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Verify the central experimental configuration.

assert RANDOM_STATE == 42
assert TEST_SIZE == 0.30

assert POSITIVE_LABEL == 1
assert NEGATIVE_LABEL == 0

assert AGE_THRESHOLD == 25
assert YOUNG_GROUP == "young"
assert OLD_GROUP == "old"

assert define_age_group(18) == YOUNG_GROUP
assert define_age_group(24) == YOUNG_GROUP
assert define_age_group(25) == OLD_GROUP
assert define_age_group(40) == OLD_GROUP

print("All project configuration checks passed.")
```

\newpage

### Notebook code cell 28

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
```

\newpage

### Notebook code cell 29

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Load OpenML German Credit dataset.
# OpenML name: credit-g

credit_data = fetch_openml(
    name="credit-g",
    version=1,
    as_frame=True,
    parser="auto",
)

raw_X = credit_data.data.copy()
raw_y = credit_data.target.copy()

print(f"Dataset name: {credit_data.details.get('name')}")
print(f"Dataset version: {credit_data.details.get('version')}")
print(f"Rows: {raw_X.shape[0]}")
print(f"Feature columns: {raw_X.shape[1]}")
print("\nTarget values:")
print(raw_y.value_counts())
```

\newpage

### Notebook code cell 31

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Encode the target.
#
# Positive class:
# good credit = 1
#
# Negative class:
# bad credit = 0

target_mapping = {
    "good": POSITIVE_LABEL,
    "bad": NEGATIVE_LABEL,
}

y = raw_y.map(target_mapping).astype(int)

print(y.value_counts().sort_index())
print(f"Positive-class rate: {y.mean():.3f}")
```

\newpage

### Notebook code cell 32

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Define the sensitive attribute from age.

X = raw_X.copy()

if "age" not in X.columns:
    raise KeyError("Expected an 'age' column in the German Credit dataset.")

X["age"] = pd.to_numeric(X["age"], errors="raise")

sensitive = X["age"].apply(define_age_group)

sensitive_counts = sensitive.value_counts().rename_axis("age_group").reset_index(name="count")
sensitive_counts
```

\newpage

### Notebook code cell 34

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Create one fixed train/test split.
#
# The same split must be reused for the baseline model and the age-removed model.

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X,
    y,
    sensitive,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

split_indices = pd.DataFrame(
    {
        "index": list(X_train.index) + list(X_test.index),
        "split": ["train"] * len(X_train) + ["test"] * len(X_test),
    }
)

split_indices_path = TABLES_DIR / "mission1_train_test_indices.csv"
split_indices.to_csv(split_indices_path, index=False)

print(f"Train rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Saved: {split_indices_path.relative_to(PROJECT_ROOT)}")
```

\newpage

### Notebook code cell 36

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
from src.mission1_utils import audit_model, build_logistic_regression_model, confusion_counts_by_group
```

\newpage

### Notebook code cell 38

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
baseline_model = build_logistic_regression_model(X_train)
baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

baseline_group_metrics, baseline_summary = audit_model(
    model_name="Baseline with age",
    y_true=y_test,
    y_pred=baseline_predictions,
    sensitive_features=sensitive_test,
)

display(baseline_group_metrics)
display(baseline_summary)
```

\newpage

### Notebook code cell 42

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Train and audit the model without age.

no_age_model = build_logistic_regression_model(X_train_no_age)
no_age_model.fit(X_train_no_age, y_train)

no_age_predictions = no_age_model.predict(X_test_no_age)

no_age_group_metrics, no_age_summary = audit_model(
    model_name="Model without age",
    y_true=y_test,
    y_pred=no_age_predictions,
    sensitive_features=sensitive_test,
)

display(no_age_group_metrics)
display(no_age_summary)
```

\newpage

### Notebook code cell 48

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def compute_chouldechova_identity(group_metrics: pd.DataFrame) -> pd.DataFrame:
    """Verify Chouldechova's identity from group confusion counts."""

    rows = []

    for _, row in group_metrics.iterrows():
        group = row["group"]

        tn = float(row["TN"])
        fp = float(row["FP"])
        fn = float(row["FN"])
        tp = float(row["TP"])

        total = tn + fp + fn + tp

        p = (tp + fn) / total
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan

        identity_rhs = (
            (p / (1 - p))
            * ((1 - ppv) / ppv)
            * (1 - fnr)
        )

        rows.append(
            {
                "group": group,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "p_base_rate": p,
                "PPV": ppv,
                "FNR": fnr,
                "observed_FPR": fpr,
                "identity_RHS": identity_rhs,
                "absolute_error": abs(fpr - identity_rhs),
            }
        )

    return pd.DataFrame(rows)


mission2_identity = compute_chouldechova_identity(baseline_group_metrics)

mission2_identity
```

\newpage

### Notebook code cell 56

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numerical and categorical columns."""

    numeric_columns = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in feature_frame.columns
        if column not in numeric_columns
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def build_logistic_regression_model(feature_frame: pd.DataFrame) -> Pipeline:
    """Build the preprocessing + logistic regression pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_frame)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model
```

\newpage

### Notebook code cell 57

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Mission 3 uses the age-removed features as predictive features.
# Age group remains available separately for fairness constraints and auditing.

X_train_m3 = X_train_no_age.copy()
X_test_m3 = X_test_no_age.copy()

sensitive_train_m3 = sensitive_train.copy()
sensitive_test_m3 = sensitive_test.copy()

y_train_m3 = y_train.copy()
y_test_m3 = y_test.copy()

m3_preprocessor = build_preprocessor(X_train_m3)

X_train_m3_transformed = m3_preprocessor.fit_transform(X_train_m3)
X_test_m3_transformed = m3_preprocessor.transform(X_test_m3)

print(f"Mission 3 training rows: {X_train_m3.shape[0]}")
print(f"Mission 3 test rows: {X_test_m3.shape[0]}")
print(f"Mission 3 feature count before preprocessing: {X_train_m3.shape[1]}")
print(f"Mission 3 transformed feature shape: {X_train_m3_transformed.shape}")
```

\newpage

### Notebook code cell 58

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def compute_reweighing_weights(
    y_values,
    sensitive_values,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute Kamiran-Calders style reweighing weights.

    Weight formula:

        w(a, y) = P(A=a) P(Y=y) / P(A=a, Y=y)

    In count form:

        w(a, y) = N_a N_y / (N N_{a,y})
    """

    y_array = np.asarray(y_values)
    sensitive_array = np.asarray(sensitive_values)

    if len(y_array) != len(sensitive_array):
        raise ValueError("y_values and sensitive_values must have the same length.")

    data = pd.DataFrame(
        {
            "group": sensitive_array,
            "label": y_array,
        }
    )

    total_n = len(data)

    group_counts = data["group"].value_counts().to_dict()
    label_counts = data["label"].value_counts().to_dict()
    joint_counts = data.groupby(["group", "label"]).size().to_dict()

    weights = []

    for group, label in zip(data["group"], data["label"]):
        numerator = group_counts[group] * label_counts[label]
        denominator = total_n * joint_counts[(group, label)]
        weights.append(numerator / denominator)

    weight_table_rows = []

    for (group, label), joint_count in sorted(joint_counts.items()):
        numerator = group_counts[group] * label_counts[label]
        denominator = total_n * joint_count
        weight = numerator / denominator

        weight_table_rows.append(
            {
                "group": group,
                "label": label,
                "group_count": group_counts[group],
                "label_count": label_counts[label],
                "joint_count": joint_count,
                "weight": weight,
            }
        )

    weight_table = pd.DataFrame(weight_table_rows)

    return np.asarray(weights, dtype=float), weight_table
```

\newpage

### Notebook code cell 59

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
reweighing_weights, reweighing_weight_table = compute_reweighing_weights(
    y_values=y_train_m3,
    sensitive_values=sensitive_train_m3,
)

reweighing_weight_table_path = TABLES_DIR / "mission3_reweighing_weights.csv"
reweighing_weight_table.to_csv(reweighing_weight_table_path, index=False)

reweighing_weight_table
```

\newpage

### Notebook code cell 60

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Reweighing mitigation.

reweighing_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
)

reweighing_model.fit(
    X_train_m3_transformed,
    y_train_m3,
    sample_weight=reweighing_weights,
)

reweighing_predictions = reweighing_model.predict(X_test_m3_transformed)

reweighing_group_metrics, reweighing_summary = audit_model(
    model_name="Reweighing",
    y_true=y_test_m3,
    y_pred=reweighing_predictions,
    sensitive_features=sensitive_test_m3,
)

display(reweighing_group_metrics)
display(reweighing_summary)
```

\newpage

### Notebook code cell 61

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def run_exponentiated_gradient(
    model_name: str,
    constraint_object,
    eps: float,
    max_iter: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, ExponentiatedGradient]:
    """Fit and audit one ExponentiatedGradient model."""

    base_estimator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint_object,
        eps=eps,
        max_iter=max_iter,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mitigator.fit(
            X_train_m3_transformed,
            y_train_m3,
            sensitive_features=sensitive_train_m3,
        )

    predictions = mitigator.predict(
        X_test_m3_transformed,
        random_state=RANDOM_STATE,
    )

    group_metrics, summary = audit_model(
        model_name=model_name,
        y_true=y_test_m3,
        y_pred=predictions,
        sensitive_features=sensitive_test_m3,
    )

    summary["eps"] = eps

    return group_metrics, summary, mitigator
```

\newpage

### Notebook code cell 62

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
eg_dp_group_metrics, eg_dp_summary, eg_dp_model = run_exponentiated_gradient(
    model_name="ExponentiatedGradient — DemographicParity",
    constraint_object=DemographicParity(),
    eps=0.02,
)

eg_eo_group_metrics, eg_eo_summary, eg_eo_model = run_exponentiated_gradient(
    model_name="ExponentiatedGradient — EqualizedOdds",
    constraint_object=EqualizedOdds(),
    eps=0.02,
)

display(eg_dp_group_metrics)
display(eg_dp_summary)

display(eg_eo_group_metrics)
display(eg_eo_summary)
```

\newpage

### Notebook code cell 63

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def threshold_optimizer_predict(
    optimizer: ThresholdOptimizer,
    features,
    sensitive_features,
):
    """Predict with ThresholdOptimizer, using random_state when supported."""

    try:
        return optimizer.predict(
            features,
            sensitive_features=sensitive_features,
            random_state=RANDOM_STATE,
        )
    except TypeError:
        return optimizer.predict(
            features,
            sensitive_features=sensitive_features,
        )


def run_threshold_optimizer(
    model_name: str,
    constraint_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, ThresholdOptimizer]:
    """Fit and audit a ThresholdOptimizer model."""

    base_estimator = build_logistic_regression_model(X_train_m3)

    optimizer = ThresholdOptimizer(
        estimator=base_estimator,
        constraints=constraint_name,
        objective="accuracy_score",
        predict_method="predict_proba",
        prefit=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.fit(
            X_train_m3,
            y_train_m3,
            sensitive_features=sensitive_train_m3,
        )

    predictions = threshold_optimizer_predict(
        optimizer,
        X_test_m3,
        sensitive_test_m3,
    )

    group_metrics, summary = audit_model(
        model_name=model_name,
        y_true=y_test_m3,
        y_pred=predictions,
        sensitive_features=sensitive_test_m3,
    )

    summary["constraint"] = constraint_name

    return group_metrics, summary, optimizer
```

\newpage

### Notebook code cell 64

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
to_dp_group_metrics, to_dp_summary, to_dp_model = run_threshold_optimizer(
    model_name="ThresholdOptimizer — DemographicParity",
    constraint_name="demographic_parity",
)

to_eo_group_metrics, to_eo_summary, to_eo_model = run_threshold_optimizer(
    model_name="ThresholdOptimizer — EqualizedOdds",
    constraint_name="equalized_odds",
)

display(to_dp_group_metrics)
display(to_dp_summary)

display(to_eo_group_metrics)
display(to_eo_summary)
```

\newpage

### Notebook code cell 65

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def group_metric_value(
    group_metrics: pd.DataFrame,
    group: str,
    metric: str,
) -> float:
    """Extract one group-specific metric value."""

    values = group_metrics.loc[group_metrics["group"] == group, metric]

    if values.empty:
        return np.nan

    return float(values.iloc[0])


def make_comparison_row(
    method: str,
    fairness_family: str,
    summary: pd.DataFrame,
    group_metrics: pd.DataFrame,
    notes: str,
) -> dict:
    """Create one row for the Mission 3 comparison table."""

    summary_row = summary.iloc[0]

    return {
        "method": method,
        "fairness_family_targeted": fairness_family,
        "accuracy": float(summary_row["accuracy"]),
        "selection_rate": float(summary_row["selection_rate"]),
        "DP_difference": float(summary_row["DP_difference"]),
        "EO_difference": float(summary_row["EO_difference"]),
        "old_TPR": group_metric_value(group_metrics, OLD_GROUP, "TPR"),
        "young_TPR": group_metric_value(group_metrics, YOUNG_GROUP, "TPR"),
        "TPR_gap_abs": abs(
            group_metric_value(group_metrics, OLD_GROUP, "TPR")
            - group_metric_value(group_metrics, YOUNG_GROUP, "TPR")
        ),
        "old_FPR": group_metric_value(group_metrics, OLD_GROUP, "FPR"),
        "young_FPR": group_metric_value(group_metrics, YOUNG_GROUP, "FPR"),
        "FPR_gap_abs": abs(
            group_metric_value(group_metrics, OLD_GROUP, "FPR")
            - group_metric_value(group_metrics, YOUNG_GROUP, "FPR")
        ),
        "old_PPV": group_metric_value(group_metrics, OLD_GROUP, "PPV"),
        "young_PPV": group_metric_value(group_metrics, YOUNG_GROUP, "PPV"),
        "notes": notes,
    }


mission3_rows = [
    make_comparison_row(
        method="Baseline without age",
        fairness_family="None",
        summary=no_age_summary,
        group_metrics=no_age_group_metrics,
        notes="Reference model for mitigation; age removed from predictive features.",
    ),
    make_comparison_row(
        method="Reweighing",
        fairness_family="Pre-processing / distribution balancing",
        summary=reweighing_summary,
        group_metrics=reweighing_group_metrics,
        notes="Changes sample weights in training.",
    ),
    make_comparison_row(
        method="ExponentiatedGradient — DemographicParity",
        fairness_family="Independence",
        summary=eg_dp_summary,
        group_metrics=eg_dp_group_metrics,
        notes="In-processing constraint targeting demographic parity.",
    ),
    make_comparison_row(
        method="ExponentiatedGradient — EqualizedOdds",
        fairness_family="Separation",
        summary=eg_eo_summary,
        group_metrics=eg_eo_group_metrics,
        notes="In-processing constraint targeting equalized odds.",
    ),
    make_comparison_row(
        method="ThresholdOptimizer — DemographicParity",
        fairness_family="Independence",
        summary=to_dp_summary,
        group_metrics=to_dp_group_metrics,
        notes="Post-processing with group-specific decision rules.",
    ),
    make_comparison_row(
        method="ThresholdOptimizer — EqualizedOdds",
        fairness_family="Separation",
        summary=to_eo_summary,
        group_metrics=to_eo_group_metrics,
        notes="Post-processing with group-specific decision rules.",
    ),
]

mission3_model_comparison = pd.DataFrame(mission3_rows)

mission3_model_comparison
```

\newpage

### Notebook code cell 66

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
mission3_model_comparison_path = TABLES_DIR / "mission3_model_comparison.csv"
mission3_model_comparison.to_csv(mission3_model_comparison_path, index=False)

print(f"Saved: {mission3_model_comparison_path.relative_to(PROJECT_ROOT)}")
```

\newpage

### Notebook code cell 68

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def run_eg_sweep() -> pd.DataFrame:
    """Run an epsilon sweep for ExponentiatedGradient.

    Important:
    We create a fresh Fairlearn constraint object for every run.
    Reusing the same constraint object across fits can cause:
    'data can be loaded only once'.
    """

    eps_values = [0.01, 0.02, 0.05, 0.10, 0.20]
    rows = []

    sweep_specs = [
        ("EG-DP", "Independence", DemographicParity),
        ("EG-EO", "Separation", EqualizedOdds),
    ]

    for short_name, family, constraint_factory in sweep_specs:
        for eps in eps_values:
            model_name = f"{short_name} eps={eps}"

            try:
                constraint_object = constraint_factory()

                group_metrics, summary, _ = run_exponentiated_gradient(
                    model_name=model_name,
                    constraint_object=constraint_object,
                    eps=eps,
                    max_iter=50,
                )

                rows.append(
                    make_comparison_row(
                        method=model_name,
                        fairness_family=family,
                        summary=summary,
                        group_metrics=group_metrics,
                        notes="Epsilon sweep point for frontier plot.",
                    )
                )

            except Exception as error:
                rows.append(
                    {
                        "method": model_name,
                        "fairness_family_targeted": family,
                        "accuracy": np.nan,
                        "selection_rate": np.nan,
                        "DP_difference": np.nan,
                        "EO_difference": np.nan,
                        "old_TPR": np.nan,
                        "young_TPR": np.nan,
                        "TPR_gap_abs": np.nan,
                        "old_FPR": np.nan,
                        "young_FPR": np.nan,
                        "FPR_gap_abs": np.nan,
                        "old_PPV": np.nan,
                        "young_PPV": np.nan,
                        "notes": f"Failed: {error}",
                    }
                )

    sweep_table = pd.DataFrame(rows)

    failed_rows = sweep_table[
        sweep_table["notes"].astype(str).str.startswith("Failed:")
    ]

    if not failed_rows.empty:
        print("Some sweep runs failed:")
        display(failed_rows[["method", "notes"]])
    else:
        print("All epsilon sweep runs completed successfully.")

    return sweep_table


mission3_eg_sweep = run_eg_sweep()

mission3_eg_sweep

mission3_eg_sweep = run_eg_sweep()

mission3_eg_sweep
```

\newpage

### Notebook code cell 71

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Combine named mitigation methods and epsilon-sweep points for plotting.

frontier_table = pd.concat(
    [
        mission3_model_comparison.assign(plot_group="main_methods"),
        mission3_eg_sweep.assign(plot_group="eg_sweep"),
    ],
    ignore_index=True,
)

frontier_table = frontier_table.dropna(
    subset=["accuracy", "DP_difference", "EO_difference"]
).copy()

frontier_table_path = TABLES_DIR / "mission3_frontier_points.csv"
frontier_table.to_csv(frontier_table_path, index=False)

frontier_table[
    [
        "method",
        "fairness_family_targeted",
        "accuracy",
        "DP_difference",
        "EO_difference",
        "TPR_gap_abs",
        "FPR_gap_abs",
    ]
]
```

\newpage

### Notebook code cell 72

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
def plot_accuracy_frontier(
    table: pd.DataFrame,
    fairness_column: str,
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    """Create and save one fairness-accuracy frontier plot."""

    plot_table = table.sort_values(fairness_column).copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        plot_table[fairness_column],
        plot_table["accuracy"],
    )

    for _, row in plot_table.iterrows():
        label = row["method"]

        if len(label) > 32:
            label = label.replace("ExponentiatedGradient", "EG")
            label = label.replace("ThresholdOptimizer", "TO")
            label = label.replace("DemographicParity", "DP")
            label = label.replace("EqualizedOdds", "EO")

        ax.annotate(
            label,
            (row[fairness_column], row["accuracy"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


dp_frontier_path = FIGURES_DIR / "mission3_dp_accuracy_frontier.png"
eo_frontier_path = FIGURES_DIR / "mission3_eo_accuracy_frontier.png"

plot_accuracy_frontier(
    table=frontier_table,
    fairness_column="DP_difference",
    output_path=dp_frontier_path,
    title="Accuracy versus Demographic-Parity Difference",
    x_label="Demographic-parity difference",
)

plot_accuracy_frontier(
    table=frontier_table,
    fairness_column="EO_difference",
    output_path=eo_frontier_path,
    title="Accuracy versus Equalized-Odds Difference",
    x_label="Equalized-odds difference",
)
```

\newpage

### Notebook code cell 74

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
# Identify best observed operating points under different criteria.
# These are not automatic normative choices. They are only aids.

best_accuracy = mission3_model_comparison.sort_values(
    "accuracy",
    ascending=False,
).iloc[0]

best_dp = mission3_model_comparison.sort_values(
    "DP_difference",
    ascending=True,
).iloc[0]

best_eo = mission3_model_comparison.sort_values(
    "EO_difference",
    ascending=True,
).iloc[0]

print("Best observed accuracy among main methods:")
display(best_accuracy.to_frame().T)

print("Smallest observed demographic-parity difference among main methods:")
display(best_dp.to_frame().T)

print("Smallest observed equalized-odds difference among main methods:")
display(best_eo.to_frame().T)
```

\newpage

### Notebook code cell 77

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
chosen_method_name = "Reweighing"

chosen_operating_point = mission3_model_comparison.loc[
    mission3_model_comparison["method"] == chosen_method_name
].copy()

if chosen_operating_point.empty:
    raise ValueError(f"Chosen method not found: {chosen_method_name}")

chosen_operating_point_path = TABLES_DIR / "mission4_chosen_operating_point.csv"
chosen_operating_point.to_csv(chosen_operating_point_path, index=False)

chosen_operating_point
```

\newpage

### Notebook code cell 78

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
baseline_no_age_row = mission3_model_comparison.loc[
    mission3_model_comparison["method"] == "Baseline without age"
].iloc[0]

reweighing_row = mission3_model_comparison.loc[
    mission3_model_comparison["method"] == "Reweighing"
].iloc[0]

mission4_delta = pd.DataFrame(
    [
        {
            "metric": "accuracy",
            "baseline_without_age": baseline_no_age_row["accuracy"],
            "chosen_reweighing": reweighing_row["accuracy"],
            "change": reweighing_row["accuracy"] - baseline_no_age_row["accuracy"],
        },
        {
            "metric": "DP_difference",
            "baseline_without_age": baseline_no_age_row["DP_difference"],
            "chosen_reweighing": reweighing_row["DP_difference"],
            "change": reweighing_row["DP_difference"] - baseline_no_age_row["DP_difference"],
        },
        {
            "metric": "EO_difference",
            "baseline_without_age": baseline_no_age_row["EO_difference"],
            "chosen_reweighing": reweighing_row["EO_difference"],
            "change": reweighing_row["EO_difference"] - baseline_no_age_row["EO_difference"],
        },
        {
            "metric": "young_TPR",
            "baseline_without_age": baseline_no_age_row["young_TPR"],
            "chosen_reweighing": reweighing_row["young_TPR"],
            "change": reweighing_row["young_TPR"] - baseline_no_age_row["young_TPR"],
        },
        {
            "metric": "young_FPR",
            "baseline_without_age": baseline_no_age_row["young_FPR"],
            "chosen_reweighing": reweighing_row["young_FPR"],
            "change": reweighing_row["young_FPR"] - baseline_no_age_row["young_FPR"],
        },
        {
            "metric": "young_PPV",
            "baseline_without_age": baseline_no_age_row["young_PPV"],
            "chosen_reweighing": reweighing_row["young_PPV"],
            "change": reweighing_row["young_PPV"] - baseline_no_age_row["young_PPV"],
        },
    ]
)

mission4_delta_path = TABLES_DIR / "mission4_chosen_delta.csv"
mission4_delta.to_csv(mission4_delta_path, index=False)

mission4_delta
```

\newpage

### Notebook code cell 81

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
mission5_obligations = pd.DataFrame(
    [
        {
            "source": "EU AI Act",
            "article_or_location": "Annex III, point 5(b)",
            "obligation_or_rule": (
                "AI systems used to evaluate the creditworthiness of natural "
                "persons or establish credit scores are high-risk, except "
                "systems used to detect financial fraud."
            ),
            "project_implementation": (
                "Treat this credit-risk classifier as high-risk if used for "
                "real creditworthiness assessment."
            ),
        },
        {
            "source": "EU AI Act",
            "article_or_location": "Article 10",
            "obligation_or_rule": (
                "High-risk AI systems trained with data require data governance, "
                "including examination of possible biases and measures to detect, "
                "prevent, and mitigate them."
            ),
            "project_implementation": (
                "Document data origin, assumptions, age-group disparities, "
                "proxy concerns, mitigation results, and subgroup performance."
            ),
        },
        {
            "source": "EU AI Act",
            "article_or_location": "Article 10(5)",
            "obligation_or_rule": (
                "Special-category personal data may exceptionally be processed "
                "for bias detection and correction when strictly necessary and "
                "subject to safeguards."
            ),
            "project_implementation": (
                "Age is not a GDPR Article 9 special category, but the same "
                "logic supports controlled retention of sensitive auditing "
                "attributes rather than blind deletion."
            ),
        },
        {
            "source": "EU AI Act",
            "article_or_location": "Article 14",
            "obligation_or_rule": (
                "High-risk AI systems must be designed so that natural persons "
                "can effectively oversee them, understand limitations, monitor "
                "operation, avoid automation bias, interpret outputs, and "
                "override or reverse outputs where appropriate."
            ),
            "project_implementation": (
                "Use the model only as decision support. A trained human reviewer "
                "must be able to override, reverse, or disregard the model output."
            ),
        },
        {
            "source": "GDPR",
            "article_or_location": "Article 22",
            "obligation_or_rule": (
                "Data subjects have rights against decisions based solely on "
                "automated processing that produce legal or similarly significant "
                "effects. Safeguards include human intervention, expression of "
                "the person's point of view, and contestation."
            ),
            "project_implementation": (
                "Do not issue final automatic rejections. Provide human review, "
                "reason codes, applicant contestation, and an appeal route."
            ),
        },
    ]
)

mission5_obligations_path = TABLES_DIR / "mission5_regulatory_obligations.csv"
mission5_obligations.to_csv(mission5_obligations_path, index=False)

mission5_obligations
```

\newpage

### Notebook code cell 82

This selected cell is included for auditability. The runnable notebook remains the authoritative executable artifact.

```python
mission5_oversight = {
    "oversight_measure": "Manual review band and appeal route",
    "which_cases_are_reviewed": (
        "All rejected applications, plus borderline approvals or rejections "
        "within a predefined score interval."
    ),
    "reviewer": "Trained credit officer",
    "reviewer_powers": (
        "The reviewer may approve, reject, request additional documents, "
        "or override or reverse the model recommendation."
    ),
    "information_given_to_reviewer": (
        "Model score, reason codes, applicant file, known model limitations, "
        "and a warning against automation bias."
    ),
    "applicant_rights": (
        "Rejected applicants receive a reason code and can request human "
        "review or contest the decision."
    ),
    "deployment_conclusion": (
        "Do not deploy as a fully automated credit-decision system. "
        "Deploy only, if at all, as a documented decision-support tool."
    ),
}

mission5_oversight_path = TABLES_DIR / "mission5_oversight_measure.json"

with mission5_oversight_path.open("w", encoding="utf-8") as file:
    json.dump(mission5_oversight, file, indent=2, ensure_ascii=False)

mission5_oversight
```



\newpage

## Appendix R — Reproduction commands

A reader can reproduce the project with the following commands.

```bash
git clone https://github.com/davisjoseph6/fairness-lab.git
cd fairness-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```



\newpage

## Appendix S — Final audit checklist

The submission satisfies the following checklist:

- The group recorded pre-computation commitments.
- The notebook runs from top to bottom.
- The sensitive attribute is age, split as young below 25 and old 25 or above.
- Good credit is encoded as the positive class.
- Mission 1 reports base rates, selection rates, TPR, FPR, PPV, demographic-parity difference, and equalized-odds difference.
- Mission 1 includes the age-removal experiment.
- Mission 2 verifies Chouldechova's identity numerically.
- Mission 3 compares Reweighing, ExponentiatedGradient, and ThresholdOptimizer.
- Mission 3 includes two separate frontier plots.
- Mission 4 chooses one operating point and defends it normatively.
- Mission 5 identifies regulatory obligations and human oversight.
- The report avoids claiming that one model is universally the fairest.
- The repository contains the notebook, HTML export, tables, figures, report, and requirements files.

## References
- Fairness in Machine Learning — Lab Brief, Prof. David Appadourai.
- Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence, Official Journal of the European Union.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng
- Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016, General Data Protection Regulation, Official Journal of the European Union.
  https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=celex%3A32016R0679
- Fairlearn documentation for MetricFrame, demographic parity, equalized odds, ExponentiatedGradient, and ThresholdOptimizer.
  https://fairlearn.org/
