"""Stage 10 — OCR layout + noise.

Converts clean Documents into a noisy copy that simulates real OCR output.
The clean copy (stored in the graph) is used for ground truth and evaluation;
the noisy copy is what the agent sees.

Noise is applied programmatically and parametrically:
- Character confusion  : common OCR mis-reads (0↔O, 1↔l, rn↔m, etc.)
- Confidence degradation: word confidence scores pulled below 1.0
- Bbox jitter          : small random offsets to polygon coordinates
- Occasional word drop : low-confidence words removed at high noise levels

`noise_level` in [0, 1] scales all effects linearly.
"""
from __future__ import annotations

import copy
from decimal import Decimal

import numpy as np

from sow_synth.models import Document, OcrLine, OcrPage, OcrWord

# ---------------------------------------------------------------------------
# Character confusion matrix — common OCR errors
# ---------------------------------------------------------------------------

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


def _confuse_char(text: str, rng: np.random.Generator, p: float) -> str:
    """Randomly apply character confusions with probability p per opportunity."""
    result = text
    for src, dst in _CONFUSIONS:
        if src in result and rng.random() < p:
            # Replace first occurrence only to keep changes subtle
            result = result.replace(src, dst, 1)
    return result


def _jitter_polygon(polygon: list[float], rng: np.random.Generator, sigma: float) -> list[float]:
    if not polygon:
        return polygon
    noise = rng.normal(0, sigma, len(polygon)).tolist()
    return [max(0.0, v + n) for v, n in zip(polygon, noise)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_noise(doc: Document, rng: np.random.Generator, noise_level: float = 0.05) -> Document:
    """Return a deep-copied Document with programmatic OCR noise applied.

    The original document is not modified.  noise_level in [0, 1].
    """
    if noise_level <= 0:
        return copy.deepcopy(doc)

    noisy_doc = copy.deepcopy(doc)
    char_p = noise_level * 0.4       # probability of confusion per opportunity
    conf_drop = noise_level * 0.25    # max confidence reduction
    bbox_sigma = noise_level * 3.0    # px std-dev for bbox jitter
    drop_p = noise_level * 0.03       # probability of dropping a word

    for page in noisy_doc.pages:
        noisy_lines: list[OcrLine] = []
        for line in page.lines:
            noisy_words: list[OcrWord] = []
            for word in line.words:
                if rng.random() < drop_p:
                    continue  # drop word entirely
                noisy_text = _confuse_char(word.text, rng, char_p)
                conf = float(np.clip(
                    word.confidence - rng.uniform(0, conf_drop),
                    0.0, 1.0,
                ))
                noisy_poly = _jitter_polygon(word.polygon, rng, bbox_sigma)
                noisy_words.append(OcrWord(text=noisy_text, confidence=conf, polygon=noisy_poly))

            # Reconstruct line text from (possibly corrupted) words
            noisy_line_text = " ".join(w.text for w in noisy_words)
            line_conf = float(np.mean([w.confidence for w in noisy_words])) if noisy_words else 0.0
            noisy_poly = _jitter_polygon(line.polygon, rng, bbox_sigma)
            noisy_lines.append(OcrLine(
                text=noisy_line_text,
                confidence=line_conf,
                polygon=noisy_poly,
                words=noisy_words,
            ))

        page.lines = noisy_lines
        # key_values are NOT noised — they represent the engine's structured
        # extraction layer, which is assumed to be more reliable than raw text.
        # Confidence degradation on key_values only:
        for kv in page.key_values:
            kv.confidence = float(np.clip(kv.confidence - rng.uniform(0, conf_drop * 0.5), 0.0, 1.0))

    return noisy_doc
