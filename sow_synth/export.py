"""Document export — one HTML per document + JSON.

Output structure:
  <out_dir>/
      index.html                    overview with links to every document
      clean/
          <doc_id>.html             all pages of the clean document in one file
          <doc_id>.json             full OCR JSON for the document
      noisy/
          <doc_id>.html
          <doc_id>.json

Each HTML file renders every page of the document stacked vertically,
separated by a visible page-break divider.  Text lines are positioned
using the polygon coordinates from the OcrPage.  Key-value extractions
appear in a panel alongside each page.
"""
from __future__ import annotations

from pathlib import Path

from sow_synth.models import Document, OcrPage

# ---------------------------------------------------------------------------
# Scale and layout constants
# ---------------------------------------------------------------------------

_PT_TO_PX   = 96.0 / 72.0   # PDF points → screen pixels
_FONT_PX    = 9.0 * _PT_TO_PX


def _px(pt: float) -> float:
    return round(pt * _PT_TO_PX, 1)


# ---------------------------------------------------------------------------
# Per-page HTML fragment
# ---------------------------------------------------------------------------

def _kv_panel(page: OcrPage) -> str:
    if not page.key_values:
        return ""
    rows = "\n".join(
        f'      <tr>'
        f'<td class="kv-key">{kv.key}</td>'
        f'<td class="kv-val{"" if kv.confidence >= 0.7 else " low"}">'
        f'{kv.value}'
        f'<span class="badge">{kv.confidence:.0%}</span>'
        f'</td></tr>'
        for kv in page.key_values
    )
    return (
        f'    <div class="kv-panel">\n'
        f'      <div class="kv-title">Extracted Fields</div>\n'
        f'      <table><tbody>\n{rows}\n      </tbody></table>\n'
        f'    </div>\n'
    )


def _page_fragment(page: OcrPage, page_num: int, total_pages: int) -> str:
    pw = _px(page.width)
    ph = _px(page.height)

    line_spans = []
    for line in page.lines:
        if not line.text.strip():
            continue
        poly = line.polygon
        if len(poly) >= 2:
            xs = poly[0::2]; ys = poly[1::2]
            x = _px(min(xs)); y = _px(min(ys))
        else:
            x, y = _px(30), _px(50)
        low = ' class="low"' if line.confidence < 0.7 else ""
        text = line.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        line_spans.append(
            f'      <span{low} style="left:{x}px;top:{y}px;">{text}</span>'
        )

    return (
        f'  <div class="page-block">\n'
        f'    <div class="page-label">Page {page_num} of {total_pages}</div>\n'
        f'    <div class="page-row">\n'
        f'      <div class="page-canvas" style="width:{pw}px;height:{ph}px;">\n'
        + "\n".join(line_spans) + "\n"
        f'      </div>\n'
        + _kv_panel(page) +
        f'    </div>\n'
        f'  </div>\n'
    )


# ---------------------------------------------------------------------------
# Full document HTML
# ---------------------------------------------------------------------------

_DOC_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{doc_id} ({doc_type})</title>
<style>
  body {{
    background: #c8c8c8;
    font-family: 'Courier New', Courier, monospace;
    font-size: {font_px}px;
    margin: 0;
    padding: 24px 16px;
    color: #111;
  }}
  h1 {{
    font-family: sans-serif;
    font-size: 15px;
    font-weight: 600;
    color: #333;
    margin: 0 0 6px 0;
  }}
  .meta {{
    font-family: sans-serif;
    font-size: 11px;
    color: #666;
    margin-bottom: 20px;
  }}
  .page-block {{
    margin-bottom: 40px;
  }}
  .page-label {{
    font-family: sans-serif;
    font-size: 11px;
    color: #555;
    margin-bottom: 6px;
    padding-left: 4px;
  }}
  .page-row {{
    display: flex;
    align-items: flex-start;
    gap: 20px;
  }}
  .page-canvas {{
    position: relative;
    background: #faf9f6;
    box-shadow: 3px 3px 14px rgba(0,0,0,.4);
    flex-shrink: 0;
    overflow: hidden;
  }}
  /* scan grain */
  .page-canvas::before {{
    content: '';
    position: absolute;
    inset: 0;
    background-image:
      url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Ccircle cx='1' cy='1' r='.4' fill='%23999' opacity='.18'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1;
  }}
  .page-canvas span {{
    position: absolute;
    white-space: pre;
    line-height: 1.0;
    z-index: 2;
  }}
  .page-canvas span.low {{
    color: #777;
    text-decoration: underline dotted #aaa;
  }}
  /* key-value panel */
  .kv-panel {{
    background: #fffff0;
    border: 1px solid #c8b860;
    border-radius: 4px;
    padding: 10px 12px;
    min-width: 220px;
    max-width: 300px;
    font-size: {kv_font_px}px;
    box-shadow: 1px 2px 6px rgba(0,0,0,.15);
    flex-shrink: 0;
  }}
  .kv-title {{
    font-family: sans-serif;
    font-size: 10px;
    font-weight: bold;
    color: #888;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: .05em;
  }}
  .kv-panel table {{
    border-collapse: collapse;
    width: 100%;
  }}
  .kv-key {{
    font-weight: bold;
    color: #444;
    padding: 2px 8px 2px 0;
    white-space: nowrap;
    vertical-align: top;
  }}
  .kv-val {{
    color: #111;
    padding: 2px 0;
    vertical-align: top;
  }}
  .kv-val.low {{ color: #b06000; }}
  .badge {{
    display: inline-block;
    font-size: 9px;
    font-family: sans-serif;
    background: #e8e8e8;
    border-radius: 2px;
    padding: 0 3px;
    margin-left: 5px;
    color: #777;
    vertical-align: middle;
  }}
  /* page-break divider */
  .page-block + .page-block {{
    border-top: 2px dashed #aaa;
    padding-top: 30px;
  }}
</style>
</head>
<body>
<h1>{doc_id}</h1>
<div class="meta">Type: {doc_type} &nbsp;|&nbsp; Pages: {n_pages} &nbsp;|&nbsp; Role: {role}</div>
"""

_DOC_HTML_FOOT = """\
</body>
</html>
"""


def _render_document_html(doc: Document) -> str:
    parts = [_DOC_HTML_HEAD.format(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type.value,
        n_pages=len(doc.pages),
        role=doc.role,
        font_px=round(_FONT_PX, 1),
        kv_font_px=round(_FONT_PX * 0.95, 1),
    )]
    for page in doc.pages:
        parts.append(_page_fragment(page, page.page_number, len(doc.pages)))
    parts.append(_DOC_HTML_FOOT)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_document(doc: Document, out_dir: Path, label: str = "") -> Path:
    """Write one HTML file + one JSON file for a Document.

    Returns the directory the files were written to.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{doc.doc_id}{('_' + label) if label else ''}"

    (out_dir / f"{slug}.html").write_text(
        _render_document_html(doc), encoding="utf-8"
    )
    (out_dir / f"{slug}.json").write_text(
        doc.model_dump_json(indent=2), encoding="utf-8"
    )
    return out_dir


def export_all(
    clean_docs: list[Document],
    noisy_docs: list[Document],
    out_dir: Path,
) -> None:
    """Export clean and noisy versions of every document, plus an index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in clean_docs:
        export_document(doc, out_dir / "clean")
    for doc in noisy_docs:
        export_document(doc, out_dir / "noisy")
    _write_index(clean_docs, noisy_docs, out_dir)


def _write_index(
    clean_docs: list[Document],
    noisy_docs: list[Document],
    out_dir: Path,
) -> None:
    noisy_by_id = {d.doc_id: d for d in noisy_docs}

    rows = []
    for doc in clean_docs:
        noisy = noisy_by_id.get(doc.doc_id)
        clean_html  = f'<a href="clean/{doc.doc_id}.html">view</a>'
        clean_json  = f'<a href="clean/{doc.doc_id}.json">json</a>'
        noisy_html  = f'<a href="noisy/{doc.doc_id}.html">view</a>' if noisy else "—"
        noisy_json  = f'<a href="noisy/{doc.doc_id}.json">json</a>'  if noisy else ""
        rows.append(
            f"<tr>"
            f"<td>{doc.doc_id}</td>"
            f"<td>{doc.doc_type.value}</td>"
            f"<td style='text-align:center'>{len(doc.pages)}</td>"
            f"<td>{clean_html} &nbsp; {clean_json}</td>"
            f"<td>{noisy_html} &nbsp; {noisy_json}</td>"
            f"</tr>"
        )

    html = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n"
        "<title>SoW Sample — Document Index</title>\n"
        "<style>\n"
        "  body{font-family:sans-serif;font-size:13px;margin:30px;}\n"
        "  h1{font-size:18px;}\n"
        "  table{border-collapse:collapse;width:100%;}\n"
        "  th,td{border:1px solid #ddd;padding:6px 10px;text-align:left;}\n"
        "  th{background:#f0f0f0;}\n"
        "  tr:nth-child(even){background:#fafafa;}\n"
        "  a{color:#1a6bb5;}\n"
        "</style>\n</head>\n<body>\n"
        f"<h1>SoW Sample &mdash; Document Index</h1>\n"
        f"<p>{len(clean_docs)} document(s)</p>\n"
        "<table>\n"
        "  <thead><tr>"
        "<th>Doc ID</th><th>Type</th><th>Pages</th>"
        "<th>Clean</th><th>Noisy</th>"
        "</tr></thead>\n"
        f"  <tbody>\n    {''.join(rows)}\n  </tbody>\n"
        "</table>\n</body>\n</html>\n"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")
