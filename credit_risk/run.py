"""Regenerate the report and the four figures."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from credit_risk.data import DATA_PATH, ROOT, load_portfolio
from credit_risk.figures import save_figures
from credit_risk.metrics import analyze
from credit_risk.report import render_report

SCREENSHOT_DIR = ROOT / "Screenshot"
REPORT_PATH = ROOT / "REPORT.md"
DOCX_PATH = ROOT / "Risk_Assessment_Report.docx"


def refresh(screenshot_dir: Path = SCREENSHOT_DIR, report_path: Path = REPORT_PATH) -> dict:
    frame = load_portfolio(DATA_PATH)
    results = analyze(frame, DATA_PATH)
    save_figures(frame, results, screenshot_dir)
    report_path.write_text(render_report(results), encoding="utf-8")
    return results


def replace_docx_images(screenshot_dir: Path = SCREENSHOT_DIR, docx_path: Path = DOCX_PATH) -> None:
    """Swap the four embedded report images for the regenerated charts."""
    if not docx_path.exists():
        return
    mapping = {
        "word/media/image1.png": screenshot_dir / "1.png",
        "word/media/image2.png": screenshot_dir / "2.png",
        "word/media/image3.png": screenshot_dir / "3.png",
        "word/media/image4.png": screenshot_dir / "4.png",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(docx_path, "r") as source, zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            replacement = mapping.get(info.filename)
            if replacement is not None:
                payload = replacement.read_bytes()
            target.writestr(info, payload)
    docx_path.write_bytes(buffer.getvalue())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate figures and REPORT.md")
    parser.add_argument("--screenshot-dir", type=Path, default=SCREENSHOT_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--update-docx-images",
        action="store_true",
        help="Replace the four images embedded in the Word report",
    )
    args = parser.parse_args(argv)
    results = refresh(args.screenshot_dir, args.report)
    if args.update_docx_images:
        replace_docx_images(args.screenshot_dir)
    print(
        f"wrote {args.report} and charts in {args.screenshot_dir} "
        f"({results['n_loans']} loans)"
    )


if __name__ == "__main__":
    main()
