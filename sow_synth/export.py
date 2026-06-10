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

Each HTML file renders every page stacked vertically with a page-break
divider.  page_text (HTML-formatted text from Azure Document Intelligence
Layout model) is embedded directly — no polygon reconstruction needed.
"""
from __future__ import annotations

from pathlib import Path

from sow_synth.models import Document, OcrPage

# ---------------------------------------------------------------------------
# Per-page HTML fragment
# ---------------------------------------------------------------------------

def _page_fragment(page: OcrPage, total_pages: int) -> str:
    return (
        f'  <div class="page-block">\n'
        f'    <div class="page-label">Page {page.page_number} of {total_pages}</div>\n'
        f'    <div class="page-canvas">\n'
        f'      {page.page_text}\n'
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
    font-size: 13px;
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
  .page-canvas {{
    background: #faf9f6;
    box-shadow: 3px 3px 14px rgba(0,0,0,.4);
    padding: 24px 32px;
    max-width: 860px;
    overflow: auto;
  }}
  .page-canvas pre {{
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'Courier New', Courier, monospace;
    font-size: 12px;
    line-height: 1.5;
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
    )]
    for page in doc.pages:
        parts.append(_page_fragment(page, len(doc.pages)))
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
        clean_html = f'<a href="clean/{doc.doc_id}.html">view</a>'
        clean_json = f'<a href="clean/{doc.doc_id}.json">json</a>'
        noisy_html = f'<a href="noisy/{doc.doc_id}.html">view</a>' if noisy else "—"
        noisy_json = f'<a href="noisy/{doc.doc_id}.json">json</a>' if noisy else ""
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
