import zipfile
from xml.etree import ElementTree as ET

from credit_risk.report import render_report
from credit_risk.run import refresh

REQUIRED = (
    "17.23%",
    "29.72%",
    "29.80%",
    "50.4%",
    "$5,280,499.32",
    "$11,734,442.82",
    "-$12,559,240.00",
    "$9,526,628.91",
    "$11,936,201.30",
    "$43,146.55",
    "$25,890.70",
    "in-sample",
    "LGD",
    "EAD",
    "0.15",
    "seed 42",
)

FORBIDDEN = (
    "2.5x higher",
    "8-12",
    "20-25%",
    "no model script",
    "annual loss of",
    "$12.6M annual loss",
    "consumer lending segment",
    "Estimated VaR at 95% confidence: $43,146.55",
    "VaR (95% confidence): $43,146.55",
    "VaR (99% confidence): $25,890.70",
    "reduce portfolio PD by",
)


def _docx_text(path) -> str:
    xml = zipfile.ZipFile(path).read("word/document.xml")
    root = ET.fromstring(xml)
    parts = []
    for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
        parts.append(node.text or "")
    return "".join(parts)


def test_report_renderer_matches_pinned_language(portfolio):
    _, results = portfolio
    text = render_report(results)
    for phrase in REQUIRED:
        assert phrase in text, phrase
    for phrase in FORBIDDEN:
        assert phrase not in text, phrase


def test_readme_matches_the_calculation():
    text = open("README.md", encoding="utf-8").read()
    for phrase in REQUIRED:
        assert phrase in text, phrase
    for phrase in FORBIDDEN:
        assert phrase not in text, phrase
    assert "python -m credit_risk" in text
    assert "requirements.txt" in text


def test_docx_matches_the_calculation():
    text = _docx_text("Risk_Assessment_Report.docx")
    for phrase in (
        "17.23%",
        "29.72%",
        "$5,280,499.32",
        "$9,526,628.91",
        "$11,936,201.30",
        "-$12,559,240.00",
        "27 of 91",
        "not credit loss",
        "not a holdout",
    ):
        assert phrase in text, phrase
    for phrase in FORBIDDEN:
        assert phrase not in text, phrase


def test_dependencies_are_pinned_and_ci_runs_them():
    lines = [
        line.strip()
        for line in open("requirements.txt", encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    for line in lines:
        name, version = line.split("==")
        assert name
        assert version
        assert not any(token in version for token in (">", "<", "*"))
    workflow = open(".github/workflows/ci.yml", encoding="utf-8").read()
    assert "requirements.txt" in workflow
    assert "pytest" in workflow
    assert 'python-version: "3.12"' in workflow


def test_one_command_writes_four_figures(tmp_path, portfolio):
    report = tmp_path / "REPORT.md"
    charts = tmp_path / "charts"
    refresh(charts, report)
    written = render_report(portfolio[1])
    assert report.read_text(encoding="utf-8") == written
    for index in range(1, 5):
        path = charts / f"{index}.png"
        assert path.stat().st_size > 5_000
        assert path.read_bytes().startswith(b"\x89PNG")
