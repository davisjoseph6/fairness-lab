from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SOURCE_REPORT = ROOT / "report" / "fairness_report.md"
EXTENDED_REPORT = ROOT / "report" / "fairness_report_extended.md"

GROUP_MEMBERS = "Davis Joseph and Malone Mfono"
INSTITUTION = "Aivancity"
PROFESSOR = "Professor David Appadourai"
REPOSITORY_URL = "https://github.com/davisjoseph6/fairness-lab.git"

ARTIFACT_LINKS = {
    "Pre-computation commitments": "https://github.com/davisjoseph6/fairness-lab/blob/main/00_commitments.md",
    "Runnable notebook": "https://github.com/davisjoseph6/fairness-lab/blob/main/notebooks/fairness_lab.ipynb",
    "Exported notebook HTML": "https://github.com/davisjoseph6/fairness-lab/blob/main/outputs/fairness_lab.html",
    "Output tables": "https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/tables",
    "Output figures": "https://github.com/davisjoseph6/fairness-lab/tree/main/outputs/figures",
    "Final report PDF": "https://github.com/davisjoseph6/fairness-lab/blob/main/report/fairness_report.pdf",
}


def read_text(relative_path: str) -> str:
    """Read a project file as UTF-8 text."""

    path = ROOT / relative_path

    if not path.exists():
        return f"[Missing file: {relative_path}]"

    return path.read_text(encoding="utf-8")


def fenced_file(
    relative_path: str,
    language: str = "text",
    max_lines: int | None = None,
) -> str:
    """Return a project file inside a Markdown fenced code block."""

    content = read_text(relative_path).strip()
    lines = content.splitlines()

    if max_lines is not None and len(lines) > max_lines:
        head_count = max_lines // 2
        tail_count = max_lines - head_count

        head = lines[:head_count]
        tail = lines[-tail_count:]
        omitted = len(lines) - len(head) - len(tail)

        lines = (
            head
            + [f"... {omitted} lines omitted; full file is in the repository ..."]
            + tail
        )

        content = "\n".join(lines)

    return f"```{language}\n{content}\n```"


def pagebreak() -> str:
    """Pandoc LaTeX page break."""

    return "\n\n\\newpage\n\n"


def section(title: str, body: str) -> str:
    """Create a page-broken level-2 Markdown section."""

    return f"{pagebreak()}## {title}\n\n{body.strip()}\n"


def strip_old_appendices_and_fix_references(text: str) -> str:
    """Remove old generated appendices and duplicate References headings."""

    if "\n## Technical appendices\n" in text:
        before = text.split("\n## Technical appendices\n", 1)[0].rstrip()

        if "\n## References\n" in text:
            references = "## References\n" + text.split("\n## References\n")[-1].strip()
            text = before + "\n\n" + references + "\n"
        else:
            text = before + "\n"

    while "## References\n\n## References" in text:
        text = text.replace("## References\n\n## References", "## References")

    return text


def ensure_submission_metadata(text: str) -> str:
    """Ensure names, institution, professor, repository, and artifact links are present."""

    old_group_line = "**Group members:** Da and Ma"

    new_metadata = f"""**Group members:** {GROUP_MEMBERS}

**Institution:** {INSTITUTION}

**Professor:** {PROFESSOR}

**GitHub repository:** {REPOSITORY_URL}"""

    if old_group_line in text:
        text = text.replace(old_group_line, new_metadata)

    if "**Group members:** Davis Joseph and Malone Mfono" not in text:
        title_marker = "# Fairness in Credit-Risk Classification\n"

        if title_marker in text:
            text = text.replace(title_marker, title_marker + "\n" + new_metadata + "\n")

    artifact_section = f"""## Repository and submission artifacts

The full project repository and submission artifacts are available here:

- Repository: {REPOSITORY_URL}
- Pre-computation commitments: {ARTIFACT_LINKS["Pre-computation commitments"]}
- Runnable notebook: {ARTIFACT_LINKS["Runnable notebook"]}
- Exported notebook HTML: {ARTIFACT_LINKS["Exported notebook HTML"]}
- Output tables: {ARTIFACT_LINKS["Output tables"]}
- Output figures: {ARTIFACT_LINKS["Output figures"]}
- Final report PDF: {ARTIFACT_LINKS["Final report PDF"]}
"""

    if "## Repository and submission artifacts" not in text:
        marker = (
            "This distinction is central to the lab: deleting a sensitive\n"
            "column from the model is not the same thing as proving that the model no\n"
            "longer produces group disparities.\n"
        )

        if marker in text:
            text = text.replace(marker, marker + "\n" + artifact_section + "\n")
        else:
            intro_marker = "\n## 1. Baseline disparity\n"

            if intro_marker not in text:
                raise RuntimeError("Could not find insertion point for artifact links.")

            before, after = text.split(intro_marker, 1)
            text = before.rstrip() + "\n\n" + artifact_section + "\n" + intro_marker + after

    return text


def ensure_references(text: str) -> str:
    """Replace the references section with the final clean references."""

    references = """## References

- Fairness in Machine Learning — Lab Brief, Prof. David Appadourai.
- Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence, Official Journal of the European Union.
  https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng
- Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016, General Data Protection Regulation, Official Journal of the European Union.
  https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=celex%3A32016R0679
- Fairlearn documentation for MetricFrame, demographic parity, equalized odds, ExponentiatedGradient, and ThresholdOptimizer.
  https://fairlearn.org/
"""

    if "\n## References\n" in text:
        before = text.split("\n## References\n", 1)[0].rstrip()
        return before + "\n\n" + references.strip() + "\n"

    return text.rstrip() + "\n\n" + references.strip() + "\n"


def insert_before_references(text: str, insertion: str) -> str:
    """Insert appendices immediately before the References section."""

    marker = "\n## References\n"

    if marker not in text:
        raise RuntimeError("Could not find ## References section.")

    before, after = text.split(marker, 1)

    return (
        before.rstrip()
        + "\n\n"
        + insertion.strip()
        + "\n\n## References\n"
        + after.strip()
        + "\n"
    )


def file_manifest() -> str:
    """Create a manifest of the submission-relevant files."""

    wanted_prefixes = [
        "00_commitments.md",
        "README.md",
        "requirements.txt",
        "requirements-lock.txt",
        "notebooks/fairness_lab.ipynb",
        "outputs/fairness_lab.html",
        "outputs/environment.json",
        "outputs/tables",
        "outputs/figures",
        "report/fairness_report.md",
        "report/fairness_report_extended.md",
        "report/fairness_report.pdf",
        "src/fairness_lab.py",
        "scripts/build_extended_report.py",
    ]

    rows: list[str] = []

    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or ".git" in path.parts or ".venv" in path.parts:
            continue

        rel = path.relative_to(ROOT).as_posix()

        if any(rel == prefix or rel.startswith(prefix + "/") for prefix in wanted_prefixes):
            rows.append(f"- `{rel}`")

    return "\n".join(rows)


def notebook_code_appendix() -> str:
    """Extract selected code cells from the notebook."""

    notebook_path = ROOT / "notebooks" / "fairness_lab.ipynb"

    if not notebook_path.exists():
        return ""

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    keywords = [
        "fetch_openml",
        "target_mapping",
        "define_age_group",
        "train_test_split",
        "build_preprocessor",
        "build_logistic_regression_model",
        "base_rate_metric",
        "ppv_metric",
        "confusion_counts_by_group",
        "audit_model",
        "compute_chouldechova_identity",
        "compute_reweighing_weights",
        "run_exponentiated_gradient",
        "run_threshold_optimizer",
        "mission3_model_comparison",
        "run_eg_sweep",
        "plot_accuracy_frontier",
        "chosen_operating_point",
        "mission5_obligations",
        "mission5_oversight",
    ]

    blocks: list[str] = []

    for idx, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))

        if not any(keyword in source for keyword in keywords):
            continue

        blocks.append(
            f"### Notebook code cell {idx}\n\n"
            "This selected cell is included for auditability. The runnable "
            "notebook remains the authoritative executable artifact.\n\n"
            f"```python\n{source.strip()}\n```"
        )

    if not blocks:
        return ""

    return section(
        "Appendix Q — Selected notebook code excerpts",
        "\n\n\\newpage\n\n".join(blocks),
    )


def build_appendices() -> str:
    """Build all appendices for the extended report."""

    appendices: list[str] = []

    appendices.append(
        section(
            "Technical appendices",
            """
The main report contains the defended fairness argument. These
appendices provide the supporting audit trail: commitments, generated
outputs, environment information, regulatory mapping, code excerpts,
and reproduction instructions.

The appendices are not meant to replace the argument. They make the
numerical and technical claims verifiable.
""",
        )
    )

    appendices.append(
        section(
            "Appendix A — Submission metadata and artifact links",
            f"""
**Group members:** {GROUP_MEMBERS}

**Institution:** {INSTITUTION}

**Professor:** {PROFESSOR}

**Repository:** {REPOSITORY_URL}

**Required artifacts:**

- Pre-computation commitments: {ARTIFACT_LINKS["Pre-computation commitments"]}
- Runnable notebook: {ARTIFACT_LINKS["Runnable notebook"]}
- Exported notebook HTML: {ARTIFACT_LINKS["Exported notebook HTML"]}
- Output tables: {ARTIFACT_LINKS["Output tables"]}
- Output figures: {ARTIFACT_LINKS["Output figures"]}
- Final report PDF: {ARTIFACT_LINKS["Final report PDF"]}
""",
        )
    )

    appendices.append(
        section(
            "Appendix B — Pre-computation commitments",
            "The full pre-computation commitment file is reproduced below.\n\n"
            + fenced_file("00_commitments.md", "markdown"),
        )
    )

    appendices.append(
        section(
            "Appendix C — Reproducibility environment",
            """
Environment metadata generated by the notebook:

"""
            + fenced_file("outputs/environment.json", "json")
            + "\n\nDirect dependency list:\n\n"
            + fenced_file("requirements.txt", "text")
            + "\n\nExact lock file used for the run:\n\n"
            + fenced_file("requirements-lock.txt", "text", max_lines=220),
        )
    )

    appendices.append(
        section(
            "Appendix D — Project file manifest",
            "The following files form the reproducible submission package.\n\n"
            + file_manifest(),
        )
    )

    appendices.append(
        section(
            "Appendix E — Mission 1: group counts and split",
            """
These tables document the sensitive-group counts and the fixed train/test
split. The same split is reused across comparable models.

### Sensitive group counts

"""
            + fenced_file("outputs/tables/mission1_sensitive_group_counts.csv", "csv")
            + "\n\n### Train/test indices excerpt\n\n"
            + fenced_file("outputs/tables/mission1_train_test_indices.csv", "csv", max_lines=260),
        )
    )

    appendices.append(
        section(
            "Appendix F — Mission 1: baseline and age-removal audit",
            """
These tables support the Mission 1 claim that young applicants were
disadvantaged under several metrics and that removing age did not remove
the disparity.

### Baseline group metrics

"""
            + fenced_file("outputs/tables/mission1_baseline_group_metrics.csv", "csv")
            + "\n\n### Baseline summary\n\n"
            + fenced_file("outputs/tables/mission1_baseline_summary.csv", "csv")
            + "\n\n### Age-removed group metrics\n\n"
            + fenced_file("outputs/tables/mission1_age_removed_group_metrics.csv", "csv")
            + "\n\n### Age-removed summary\n\n"
            + fenced_file("outputs/tables/mission1_age_removed_summary.csv", "csv")
            + "\n\n### Mission 1 model comparison\n\n"
            + fenced_file("outputs/tables/mission1_model_comparison.csv", "csv"),
        )
    )

    appendices.append(
        section(
            "Appendix G — Mission 1 interpretation matrix",
            """
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
""",
        )
    )

    appendices.append(
        section(
            "Appendix H — Mission 2: Chouldechova identity outputs",
            """
This appendix records the numerical verification of the identity:

FPR = (p / (1 - p)) × ((1 - PPV) / PPV) × (1 - FNR)

The identity reproduces the observed FPR in both groups.

### Identity table

"""
            + fenced_file("outputs/tables/mission2_chouldechova_identity.csv", "csv")
            + "\n\n### Equal-PPV counterfactual\n\n"
            + fenced_file("outputs/tables/mission2_equal_ppv_counterfactual.csv", "csv"),
        )
    )

    appendices.append(
        section(
            "Appendix I — Mission 2 interpretation",
            """
The old group had a higher base rate than the young group. When the
groups have different base rates, equal PPV and equal FPR generally
cannot both be imposed in ordinary non-perfect conditions.

The counterfactual used the pooled PPV as a common PPV value. Under that
common PPV, the identity forced the FPR values far apart. This is the
numerical form of the impossibility result used in the report.

The key point is that the conflict is not a programming error. It is a
counting relationship involving prevalence, predictive value, false
negative rate, and false positive rate.
""",
        )
    )

    appendices.append(
        section(
            "Appendix J — Mission 3: reweighing and mitigation setup",
            """
Mission 3 used age-removed predictive features. The age group remained
available separately for auditing and for fairness constraints.

Reweighing changes the effective training distribution by giving
different weights to group-label combinations. ExponentiatedGradient
imposes an in-processing fairness constraint. ThresholdOptimizer changes
post-processing decision thresholds and may use group-specific decision
rules.

### Reweighing weights

"""
            + fenced_file("outputs/tables/mission3_reweighing_weights.csv", "csv"),
        )
    )

    appendices.append(
        section(
            "Appendix K — Mission 3: main mitigation comparison",
            """
The following comparison table is the main quantitative basis for the
Mission 3 discussion.

"""
            + fenced_file("outputs/tables/mission3_model_comparison.csv", "csv"),
        )
    )

    appendices.append(
        section(
            "Appendix L — Mission 3: epsilon sweep and frontier data",
            """
The epsilon sweep provides additional operating points for the
fairness-accuracy frontier.

### ExponentiatedGradient epsilon sweep

"""
            + fenced_file("outputs/tables/mission3_eg_epsilon_sweep.csv", "csv")
            + "\n\n### Frontier points\n\n"
            + fenced_file("outputs/tables/mission3_frontier_points.csv", "csv"),
        )
    )

    appendices.append(
        section(
            "Appendix M — Mission 3 frontier figures",
            """
The two figures below are repeated in the appendix for readability. They
should be interpreted as two different frontiers, not as one generic
fairness score.

### Accuracy versus demographic-parity difference

![Accuracy versus demographic-parity difference](../outputs/figures/mission3_dp_accuracy_frontier.png)

\\newpage

### Accuracy versus equalized-odds difference

![Accuracy versus equalized-odds difference](../outputs/figures/mission3_eo_accuracy_frontier.png)
""",
        )
    )

    appendices.append(
        section(
            "Appendix N — Mission 4: chosen operating point",
            """
The chosen operating point is Reweighing. The group chose it because it
best matched the stated normative priority: reducing unequal error rates
and false-rejection burden without reducing accuracy.

### Chosen operating point

"""
            + fenced_file("outputs/tables/mission4_chosen_operating_point.csv", "csv")
            + "\n\n### Delta from baseline without age\n\n"
            + fenced_file("outputs/tables/mission4_chosen_delta.csv", "csv")
            + "\n\n### Chosen summary\n\n"
            + fenced_file("outputs/tables/mission4_chosen_summary.json", "json"),
        )
    )

    appendices.append(
        section(
            "Appendix O — Mission 5: regulatory mapping",
            """
This appendix records the legal and deployment mapping used in Mission
5. The report conclusion is that the system should not be deployed as a
fully automated credit-decision system.

### Regulatory obligations

"""
            + fenced_file("outputs/tables/mission5_regulatory_obligations.csv", "csv")
            + "\n\n### Oversight measure\n\n"
            + fenced_file("outputs/tables/mission5_oversight_measure.json", "json"),
        )
    )

    appendices.append(
        section(
            "Appendix P — Shared Python module",
            """
The shared Python module contains the reproducibility configuration and
project utilities.

"""
            + fenced_file("src/fairness_lab.py", "python"),
        )
    )

    code_appendix = notebook_code_appendix()
    if code_appendix:
        appendices.append(code_appendix)

    appendices.append(
            section(
                "Appendix R — Reproduction commands",
                """
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
"""
)
)

    appendices.append(
        section(
            "Appendix S — Final audit checklist",
            """
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
"""
        )
    )

    return "\n".join(appendices)


def main() -> None:
    if not SOURCE_REPORT.exists():
        raise FileNotFoundError(SOURCE_REPORT)

    text = SOURCE_REPORT.read_text(encoding="utf-8")

    text = strip_old_appendices_and_fix_references(text)
    text = ensure_submission_metadata(text)
    text = ensure_references(text)

    appendices = build_appendices()
    extended = insert_before_references(text, appendices)

    while "## References\n\n## References" in extended:
        extended = extended.replace("## References\n\n## References", "## References")

    EXTENDED_REPORT.write_text(extended, encoding="utf-8")

    print(f"Wrote: {EXTENDED_REPORT}")
    print(f"Characters: {len(extended):,}")


if __name__ == "__main__":
    main()
