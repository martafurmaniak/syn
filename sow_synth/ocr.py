"""Stage 10 — OCR noise.

Applies programmatic, parameterised noise to the HTML page_text of each page,
simulating real OCR errors.  The clean copy (in the graph) is kept for ground
truth; the noisy copy is what the agent sees.

Noise types (all scaled by noise_level in [0, 1]):
  Character confusion  — common OCR mis-reads applied to text nodes
                         (0<->O, 1<->l, rn<->m, etc.)
  Word drop            — occasional words removed from the text
"""
from __future__ import annotations

import copy
import re

import numpy as np

from sow_synth.models import Document, OcrPage

# Character confusion pairs — common OCR errors
_CONFUSIONS: list[tuple[str, str]] = [
    ("0", "O"), ("O", "0"),
    ("1", "l"), ("l", "1"),
    ("I", "l"), ("l", "I"),
    ("rn", "m"), ("m", "rn"),
    ("vv", "w"), ("w", "vv"),
    ("5", "S"), ("S", "5"),
    ("8", "B"), ("B", "8"),
    ("6", "G"), ("G", "6"),
]

# Matches HTML tags so we can skip them during noise application
_TAG_RE = re.compile(r"(<[^>]+>)")


def _confuse_text(text: str, rng: np.random.Generator, char_p: float) -> str:
    """Apply character confusions to text, skipping HTML tags."""
    parts = _TAG_RE.split(text)
    result = []
    for part in parts:
        if _TAG_RE.fullmatch(part):
            result.append(part)   # HTML tag — leave untouched
        else:
            for src, dst in _CONFUSIONS:
                if src in part and rng.random() < char_p:
                    part = part.replace(src, dst, 1)
            result.append(part)
    return "".join(result)


def _drop_words(text: str, rng: np.random.Generator, drop_p: float) -> str:
    """Randomly remove individual words from text nodes, preserving HTML tags."""
    parts = _TAG_RE.split(text)
    result = []
    for part in parts:
        if _TAG_RE.fullmatch(part):
            result.append(part)
        else:
            words = part.split(" ")
            kept = [w for w in words if not w or rng.random() >= drop_p]
            result.append(" ".join(kept))
    return "".join(result)


def _noise_page(page: OcrPage, rng: np.random.Generator,
                char_p: float, drop_p: float) -> OcrPage:
    text = page.page_text
    text = _confuse_text(text, rng, char_p)
    text = _drop_words(text, rng, drop_p)
    return OcrPage(page_number=page.page_number, page_text=text)


def apply_noise(doc: Document, rng: np.random.Generator,
                noise_level: float = 0.05) -> Document:
    """Return a deep-copied Document with programmatic OCR noise applied.

    The original document is not modified.  noise_level in [0, 1].
    """
    if noise_level <= 0:
        return copy.deepcopy(doc)

    char_p = noise_level * 0.4
    drop_p = noise_level * 0.03

    noisy = copy.deepcopy(doc)
    noisy.pages = [_noise_page(p, rng, char_p, drop_p) for p in noisy.pages]
    return noisy
