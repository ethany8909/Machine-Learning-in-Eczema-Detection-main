"""Typeset MANUSCRIPT.md into a scientific-journal-style PDF.

Uses ReportLab (Times serif, justified body, 1-inch margins, US Letter) with
running header and page numbers. Handles headings, paragraphs with inline
bold/italic, markdown pipe tables, embedded raster figures, SVG figures
(via svglib), bullet lists, and horizontal rules.

    python build_manuscript_pdf.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "MANUSCRIPT.md"
OUT = ROOT / "MANUSCRIPT.pdf"

PAGE_W, PAGE_H = LETTER
MARGIN = 1.0 * inch
FRAME_W = PAGE_W - 2 * MARGIN
RUNNING_HEAD = "Fairness-Aware Multimodal Eczema-Psoriasis Differentiation"

# --------------------------------------------------------------------------- #
# Character sanitation: Times base-14 uses WinAnsi; map anything outside it.
# --------------------------------------------------------------------------- #
CHAR_MAP = {
    "\u2192": " to ", "\u2190": " from ", "\u2194": "-",
    "\u2248": "~", "\u2264": "<=", "\u2265": ">=", "\u2260": "!=",
    "\u03b1": "alpha", "\u03b2": "beta", "\u03c1": "rho", "\u03bb": "lambda",
    "\u2032": "'", "\u2033": '"', "\u200b": "", "\ufeff": "",
    "\u2026": "...", "\u22c5": ".", "\u00d7": "x",
}


def sanitize(s: str) -> str:
    for bad, good in CHAR_MAP.items():
        s = s.replace(bad, good)
    # drop anything still unrepresentable in WinAnsi
    return s.encode("cp1252", "ignore").decode("cp1252")


def inline(s: str) -> str:
    """Markdown inline -> ReportLab mini-HTML."""
    s = sanitize(s)
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", s)
    return s


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
S = {
    "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=16, leading=20,
                            alignment=TA_CENTER, spaceAfter=14),
    "authors": ParagraphStyle("authors", fontName="Times-Roman", fontSize=11, leading=15,
                              alignment=TA_CENTER, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=13, leading=16,
                         spaceBefore=16, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Times-BoldItalic", fontSize=11.5, leading=14,
                         spaceBefore=11, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=14.5,
                           alignment=TA_JUSTIFY, spaceAfter=8, firstLineIndent=0),
    "abstract": ParagraphStyle("abstract", fontName="Times-Roman", fontSize=10, leading=13.5,
                               alignment=TA_JUSTIFY, spaceAfter=7,
                               leftIndent=14, rightIndent=14),
    "caption": ParagraphStyle("caption", fontName="Times-Roman", fontSize=9, leading=12,
                              alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Times-Roman", fontSize=10.5, leading=14,
                             alignment=TA_JUSTIFY, leftIndent=16, bulletIndent=5,
                             spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.4, leading=9),
    "cellh": ParagraphStyle("cellh", fontName="Times-Bold", fontSize=7.4, leading=9),
}


def page_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8.5)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(MARGIN, PAGE_H - 0.62 * inch, RUNNING_HEAD)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.62 * inch, "Preprint")
    canvas.setStrokeColor(colors.HexColor("#999999"))
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, PAGE_H - 0.70 * inch, PAGE_W - MARGIN, PAGE_H - 0.70 * inch)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(PAGE_W / 2, 0.6 * inch, str(canvas.getPageNumber()))
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_table(rows):
    """rows: list of list[str] (first row = header)."""
    header, body = rows[0], rows[1:]
    ncol = len(header)
    data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
    for r in body:
        data.append([Paragraph(inline(c), S["cell"]) for c in r])

    # width proportional to max content length, clamped
    lens = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(ncol)]
    lens = [max(4, min(l, 34)) for l in lens]
    total = sum(lens)
    widths = [FRAME_W * l / total for l in lens]

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_image(path: Path):
    max_w, max_h = FRAME_W, 6.1 * inch
    if path.suffix.lower() == ".svg":
        from svglib.svglib import svg2rlg
        d = svg2rlg(str(path))
        if d is None:
            return None
        sc = min(max_w / d.width, max_h / d.height, 1.0)
        d.width, d.height = d.width * sc, d.height * sc
        d.scale(sc, sc)
        d.hAlign = "CENTER"
        return d
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        iw, ih = im.size
    sc = min(max_w / iw, max_h / ih)
    img = Image(str(path), width=iw * sc, height=ih * sc)
    img.hAlign = "CENTER"
    return img


def parse(md: str):
    flow = []
    lines = md.split("\n")
    i = 0
    in_abstract = False
    section = ""
    para_buf: list[str] = []

    def flush(style_key="body"):
        nonlocal para_buf
        if para_buf:
            txt = " ".join(para_buf).strip()
            if txt:
                flow.append(Paragraph(inline(txt), S[style_key]))
            para_buf = []

    while i < len(lines):
        ln = lines[i].rstrip()
        stripped = ln.strip()

        # --- blank line ends a paragraph
        if not stripped:
            flush("abstract" if in_abstract else "body")
            i += 1
            continue

        # --- horizontal rule: ignore (section spacing handled by styles)
        if re.fullmatch(r"-{3,}", stripped):
            flush("abstract" if in_abstract else "body")
            i += 1
            continue

        # --- headings
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            flush("abstract" if in_abstract else "body")
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                flow.append(Paragraph(inline(text), S["title"]))
            else:
                if level == 2:
                    section = text.strip().lower()
                in_abstract = (text.strip().lower() == "abstract")
                flow.append(Paragraph(inline(text), S["h1" if level == 2 else "h2"]))
            i += 1
            continue

        # --- image (bare, i.e. not already consumed by a figure caption)
        m = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", stripped)
        if m:
            flush("abstract" if in_abstract else "body")
            p = (ROOT / m.group(1)).resolve()
            if p.exists():
                obj = build_image(p)
                if obj is not None:
                    flow.append(Spacer(1, 4))
                    flow.append(obj)
                    flow.append(Spacer(1, 10))
            i += 1
            continue

        # --- table block
        if stripped.startswith("|"):
            flush("abstract" if in_abstract else "body")
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = lines[i].strip().strip("|")
                cells = [c.strip() for c in raw.split("|")]
                if not re.fullmatch(r"[\s:\-|]+", lines[i].strip()):
                    rows.append(cells)
                i += 1
            if rows:
                flow.append(Spacer(1, 3))
                flow.append(build_table(rows))
                flow.append(Spacer(1, 10))
            continue

        # --- bullet
        if stripped.startswith("- "):
            flush("abstract" if in_abstract else "body")
            flow.append(Paragraph(inline(stripped[2:]), S["bullet"], bulletText="\u2022"))
            i += 1
            continue

        # --- caption lines (Table N. / Figure N.)
        mcap = re.match(r"^\*\*(Table|Figure)\s", stripped)
        if mcap:
            flush("abstract" if in_abstract else "body")
            kind = mcap.group(1)
            buf = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("|", "!", "#")):
                buf.append(lines[i].strip())
                i += 1
            cap = Paragraph(inline(" ".join(buf)), S["caption"])

            # Figure captions go BELOW their image, bound to it so they never split.
            if kind == "Figure":
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                mimg = re.match(r"^!\[[^\]]*\]\(([^)]+)\)\s*$", lines[j].strip()) if j < len(lines) else None
                if mimg:
                    p = (ROOT / mimg.group(1)).resolve()
                    obj = build_image(p) if p.exists() else None
                    if obj is not None:
                        flow.append(Spacer(1, 6))
                        flow.append(KeepTogether([obj, Spacer(1, 5), cap]))
                        flow.append(Spacer(1, 8))
                        i = j + 1
                        continue
            # Table captions stay above their table.
            flow.append(cap)
            continue

        # --- author block (immediately after title, before first ---)
        if stripped.startswith("**Authors:**") or stripped.startswith("**Corresponding Author:**") \
                or re.match(r"^[\u00b9\u00b2\[]", stripped):
            flush()
            flow.append(Paragraph(inline(stripped), S["authors"]))
            i += 1
            continue

        # --- abbreviation / definition lines keep their own line break
        if section == "abbreviations" and re.match(r"^[A-Za-z0-9\-/ ]{2,24}:\s", stripped):
            flush()
            flow.append(Paragraph(inline(stripped), S["body"]))
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    flush("abstract" if in_abstract else "body")
    return flow


def main():
    md = SRC.read_text(encoding="utf-8")
    story = parse(md)

    doc = BaseDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.95 * inch, bottomMargin=0.9 * inch,
        title="Fairness-Aware Multimodal Eczema-Psoriasis Differentiation",
        author="[Author 1]", subject="Algorithm Development and Validation",
    )
    frame = Frame(MARGIN, 0.9 * inch, FRAME_W, PAGE_H - 0.95 * inch - 0.9 * inch, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_furniture)])
    doc.build(story)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
