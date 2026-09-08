"""
Builds the downloadable per-query report (PDF) — query, answer, confidence,
execution trace, tools used. Uses fpdf2 to keep the dependency light.

Two bugs fixed here (both were live crashes, not hypothetical):

1. THE 500 ERROR ("Not enough horizontal space to render a single character"):
   fpdf2's multi_cell() defaults to new_x=XPos.RIGHT. When a call didn't need
   to wrap (e.g. a short "Report ID: ..." line), the cursor was left sitting
   at the right edge of the page instead of returning to the left margin.
   The *next* multi_cell() call then had ~10mm of width to work with instead
   of the full page, and blew up on the first character that didn't fit.
   Fix: every multi_cell() call below explicitly passes
   new_x=XPos.LMARGIN, new_y=YPos.NEXT so the cursor always resets.

2. UNICODE / MULTILINGUAL CRASH (latent, not yet hit but guaranteed to fire):
   the PS requires the system to answer in the query's language, including
   Indian regional languages. Core "Helvetica" is Latin-1 only, so a Tamil
   or Hindi query/answer would raise FPDFUnicodeEncodingException the moment
   report generation is exercised end-to-end. Fix: embed DejaVuSans (broad
   Latin/symbol coverage) as the base font, with Noto Sans Tamil and Noto
   Sans Devanagari registered as automatic fallback fonts for characters
   outside DejaVuSans's coverage.
"""
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from app.config import REPORT_DIR

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"


def _build_pdf() -> FPDF:
    """Construct an FPDF instance with Unicode + Indian-script fallback fonts loaded."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_font("DejaVu", "", str(FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("NotoTamil", "", str(FONT_DIR / "NotoSansTamil-Regular.ttf"))
    pdf.add_font("NotoDevanagari", "", str(FONT_DIR / "NotoSansDevanagari-Regular.ttf"))

    # If a character isn't in DejaVu (e.g. Tamil/Hindi script), fpdf2 will
    # automatically retry it against these fonts in order.
    pdf.set_fallback_fonts(["NotoTamil", "NotoDevanagari"])
    return pdf


def _mc(pdf: FPDF, h: float, text: str) -> None:
    """multi_cell wrapper that always resets the cursor to the left margin.
    This is the actual fix for the 500 error — see module docstring."""
    pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def build_pdf_report(report_id: str, query: str, answer: str, confidence: float,
                      tools_used: list[str], execution_trace: list[dict]) -> Path:
    pdf = _build_pdf()

    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "SatQuery AI - Execution Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("DejaVu", "", 11)
    _mc(pdf, 7, f"Report ID: {report_id}")
    _mc(pdf, 7, f"Query: {query}")
    _mc(pdf, 7, f"Analysis Modules: {', '.join(tools_used)}")

    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Answer", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 11)
    _mc(pdf, 7, answer)

    pdf.ln(4)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Operational Execution Trace", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("DejaVu", "", 10)
    for step in execution_trace:
        step_name = step.get('step', '').replace('_', ' ').title()
        _mc(pdf, 6, f"- {step_name}: {step.get('detail', '')}")

    out_path = REPORT_DIR / f"{report_id}.pdf"
    pdf.output(str(out_path))
    return out_path
