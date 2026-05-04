"""
teams_caption_capture.py
========================
Continuous capture of live captions from Microsoft Teams meetings via OCR
screen-scraping. Runs entirely locally. No Graph API / Copilot dependency.

Pipeline
--------
    screen region (mss)
        -> preprocess (numpy)
            -> OCR (Tesseract or EasyOCR)
                -> CaptionDeduplicator (scroll-aware diffing)
                    -> TranscriptWriter (append-flush to .txt / .md)

Dependencies
------------
Core:
    pip install mss numpy pillow

One of:
    pip install pytesseract   (lighter, needs Tesseract binary installed)
    pip install easyocr       (heavier but more accurate, pip-only)

Tesseract binary (Windows):
    https://github.com/UB-Mannheim/tesseract/wiki
    If it is not on PATH, set the full path near the top of TesseractEngine.

Usage
-----
    # First run - interactive calibrator:
    python teams_caption_capture.py

    # Subsequent runs - reuse coordinates:
    python teams_caption_capture.py --bbox 340,820,1240,180

    # Choose engine / output:
    python teams_caption_capture.py --engine easyocr --output my_meeting.md
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import mss
from PIL import Image


# ============================ Configuration =================================

DEFAULT_POLL_INTERVAL = 1.5           # seconds between captures
DEFAULT_OUTPUT_DIR = Path("./transcripts")
TAIL_WINDOW_WORDS = 80                # committed words kept in RAM for diffing
MIN_FUZZY_OVERLAP = 2                 # words needed for a "trusted" overlap
FRESH_APPEND_MIN_WORDS = 3            # min words before we trust a no-overlap OCR
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


# ======================== Windows DPI awareness =============================

def _enable_dpi_awareness() -> None:
    """
    Makes tkinter coords and mss coords agree on Windows when display scaling
    is != 100%. Without this, a box drawn by the calibrator grabs the wrong
    pixels on HiDPI laptops.
    """
    if sys.platform != "win32":
        return
    try:
        from ctypes import windll
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            from ctypes import windll
            windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# =========================== Calibration helper =============================

def calibrate_bounding_box() -> Tuple[int, int, int, int]:
    """
    Opens a translucent fullscreen overlay. The user click-drags a rectangle
    over the Teams caption strip; on mouse release we return
    (left, top, width, height) in PHYSICAL screen pixels.

    ESC cancels.
    """
    import tkinter as tk

    result: dict = {}

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.30)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.title("Teams Caption Calibrator")

    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    hint = tk.Label(
        root,
        text=("Drag a rectangle around the Teams caption area, then release. "
              "ESC to cancel."),
        bg="#ffeb3b", fg="black", font=("Segoe UI", 14, "bold"), padx=12, pady=6,
    )
    hint.place(relx=0.5, y=24, anchor="n")

    start = {"x": 0, "y": 0}
    rect = {"id": None}

    def on_press(e):
        start["x"], start["y"] = e.x_root, e.y_root
        if rect["id"] is not None:
            canvas.delete(rect["id"])
        rect["id"] = canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#ff1744", width=3,
        )

    def on_drag(e):
        # Translate root coords back into canvas coords
        x1 = start["x"] - root.winfo_rootx()
        y1 = start["y"] - root.winfo_rooty()
        canvas.coords(rect["id"], x1, y1, e.x, e.y)

    def on_release(e):
        x1, y1 = start["x"], start["y"]
        x2, y2 = e.x_root, e.y_root
        left, top = min(x1, x2), min(y1, y2)
        width, height = abs(x2 - x1), abs(y2 - y1)
        result["bbox"] = (left, top, width, height)
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", lambda _: root.destroy())
    root.mainloop()

    bbox = result.get("bbox")
    if not bbox or bbox[2] < 20 or bbox[3] < 10:
        raise RuntimeError("Calibration cancelled or bounding box too small.")
    return bbox


# ============================== OCR engines =================================

class OCREngine:
    """Abstract OCR backend."""
    def read(self, img: np.ndarray) -> str:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    """Fast, lightweight OCR via the Tesseract binary."""

    def __init__(self, lang: str = "eng", tesseract_cmd: Optional[str] = None):
        import pytesseract
        self._pt = pytesseract
        if tesseract_cmd:
            self._pt.pytesseract.tesseract_cmd = tesseract_cmd
        self.lang = lang
        # --psm 6 = assume a uniform block of text (matches caption strips well).
        self.config = "--oem 3 --psm 6"

    def read(self, img: np.ndarray) -> str:
        pil = Image.fromarray(img)
        # Upscale 2x: caption text is small; Tesseract likes ~30+ px glyphs.
        w, h = pil.size
        pil = pil.resize((w * 2, h * 2), Image.LANCZOS)
        try:
            text = self._pt.image_to_string(pil, lang=self.lang, config=self.config)
        except Exception as e:
            logging.warning("Tesseract failed: %s", e)
            return ""
        return text.strip()


class EasyOCREngine(OCREngine):
    """More accurate, pip-only OCR (downloads ~100MB model on first run)."""

    def __init__(self, lang_list: Optional[List[str]] = None, gpu: bool = False):
        import easyocr
        self.reader = easyocr.Reader(lang_list or ["en"], gpu=gpu, verbose=False)

    def read(self, img: np.ndarray) -> str:
        try:
            # paragraph=True merges line fragments into natural reading order.
            lines = self.reader.readtext(img, detail=0, paragraph=True)
        except Exception as e:
            logging.warning("EasyOCR failed: %s", e)
            return ""
        return " ".join(lines).strip()


# ============================ Preprocessing =================================

def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """
    BGRA screen grab -> grayscale, auto-inverted for OCR.

    Teams captions are white-on-dark-semitransparent by default, but themes vary.
    We convert to grayscale and, if the image is mostly dark, invert so glyphs
    end up dark-on-light (what OCR engines expect).
    """
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # Standard BT.601 luminance (BGR order from mss)
    gray = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2])
    gray = gray.astype(np.uint8)
    if gray.mean() < 128:
        gray = 255 - gray
    return gray


# ============================ Deduplication =================================

_WORD_RE = re.compile(r"\S+")

def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text)

def _normalize(w: str) -> str:
    """Lowercase + strip punctuation for forgiving comparison."""
    return re.sub(r"[^\w]", "", w).lower()


class CaptionDeduplicator:
    """
    Incrementally builds a committed transcript from overlapping OCR reads.

    Why this is needed
    ------------------
    Teams captions are a scrolling window: each 1.5s OCR read typically contains
    the TAIL of text we already captured plus a few new words at the end.
    Naive concatenation would duplicate everything; naive "only append when
    text changes" would miss words when the buffer scrolls.

    Algorithm
    ---------
    On each new OCR reading `new`:

    1. Tokenize both the committed transcript tail and `new` into words, and
       normalize (lowercase, strip punct) for tolerant comparison.

    2. PASS 1 - Strict suffix/prefix overlap.
       Find the largest k such that tail[-k:] == new[:k]. This is the common
       case when captions scroll cleanly (no OCR noise): we append new[k:].

    3. PASS 2 - Fuzzy longest common block (difflib.SequenceMatcher).
       If strict overlap is weak (< 2 words), fall back to the longest matching
       contiguous block anywhere inside tail vs anywhere inside new. This
       tolerates one-word OCR drops / misreads inside the overlap region.
       We append whatever comes AFTER the matched block in `new`.

    4. NO overlap -> treat as a fresh speaker / caption jump.
       Only commit if `new` is substantial (>= FRESH_APPEND_MIN_WORDS) to avoid
       noisy single-word blips.

    Edge behavior: identical consecutive reads produce nothing (overlap == full
    length, nothing after it to append).
    """

    def __init__(self, tail_window: int = TAIL_WINDOW_WORDS):
        self.tail_window = tail_window
        self.committed: List[str] = []

    def _tail(self) -> List[str]:
        return self.committed[-self.tail_window:]

    def ingest(self, raw_text: str) -> Optional[str]:
        """Feed one OCR reading. Returns newly appended text (or None)."""
        new_words = [w for w in _tokenize(raw_text) if w]
        if not new_words:
            return None

        # Cold start
        if not self.committed:
            self.committed.extend(new_words)
            return " ".join(new_words)

        tail = self._tail()
        tail_n = [_normalize(w) for w in tail]
        new_n = [_normalize(w) for w in new_words]

        # ---- Pass 1: strict suffix-prefix overlap ----
        max_k = min(len(tail_n), len(new_n))
        overlap = 0
        for k in range(max_k, 0, -1):
            if tail_n[-k:] == new_n[:k]:
                overlap = k
                break

        if overlap >= MIN_FUZZY_OVERLAP:
            to_append = new_words[overlap:]
        else:
            # ---- Pass 2: fuzzy longest common block ----
            matcher = SequenceMatcher(a=tail_n, b=new_n, autojunk=False)
            m = matcher.find_longest_match(0, len(tail_n), 0, len(new_n))
            trusted = (
                m.size >= MIN_FUZZY_OVERLAP
                or (m.size == 1 and len(tail_n[m.a]) >= 5)  # one long word is fine
            )
            if trusted:
                to_append = new_words[m.b + m.size:]
            elif len(new_words) >= FRESH_APPEND_MIN_WORDS:
                # No overlap + substantial text -> new speaker / scrolled past buffer
                to_append = new_words
            else:
                # Probably OCR noise
                return None

        if not to_append:
            return None
        self.committed.extend(to_append)
        return " ".join(to_append)

    @property
    def full_transcript(self) -> str:
        return " ".join(self.committed)


# ============================= Transcript I/O ===============================

class TranscriptWriter:
    """Append-and-flush writer so nothing is lost if the script crashes."""

    def __init__(self, path: Path, as_markdown: bool = False):
        self.path = path
        self.as_markdown = as_markdown
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().isoformat(timespec="seconds")
        header = (
            f"# Teams Caption Transcript\n\n_Started: {ts}_\n\n"
            if as_markdown
            else f"Teams Caption Transcript  (started {ts})\n\n"
        )
        self.path.write_text(header, encoding="utf-8")

    def append(self, text: str) -> None:
        if not text.strip():
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text + " ")
            f.flush()

    def newline(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.flush()


# ============================== Capture loop ================================

def capture_loop(
    bbox: Tuple[int, int, int, int],
    ocr: OCREngine,
    writer: TranscriptWriter,
    dedup: CaptionDeduplicator,
    interval: float = DEFAULT_POLL_INTERVAL,
) -> None:
    left, top, width, height = bbox
    region = {"left": left, "top": top, "width": width, "height": height}

    logging.info("Capturing %s every %.1fs. Ctrl+C to stop.", region, interval)

    empty_streak = 0
    with mss.mss() as sct:
        try:
            while True:
                t0 = time.time()
                try:
                    shot = np.array(sct.grab(region))
                    pre = preprocess_for_ocr(shot)
                    text = ocr.read(pre)
                except Exception as e:
                    logging.warning("Capture/OCR error (continuing): %s", e)
                    text = ""

                if text:
                    empty_streak = 0
                    new_chunk = dedup.ingest(text)
                    if new_chunk:
                        logging.info("+ %s", new_chunk)
                        writer.append(new_chunk)
                else:
                    empty_streak += 1
                    if empty_streak in (5, 20, 60):
                        logging.debug("No captions detected for %d cycles.",
                                      empty_streak)

                elapsed = time.time() - t0
                time.sleep(max(0.0, interval - elapsed))
        except KeyboardInterrupt:
            logging.info("Stopped by user.")


# ================================== CLI =====================================

def _parse_bbox(s: str) -> Tuple[int, int, int, int]:
    try:
        parts = [int(p.strip()) for p in s.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("bbox must be four integers") from e
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox format: X,Y,WIDTH,HEIGHT")
    return tuple(parts)  # type: ignore[return-value]


def main() -> None:
    _enable_dpi_awareness()

    parser = argparse.ArgumentParser(
        description="Capture MS Teams live captions via local OCR screen-scraping."
    )
    parser.add_argument(
        "--bbox", type=_parse_bbox,
        help="Caption region as X,Y,WIDTH,HEIGHT. Omit to launch the calibrator.",
    )
    parser.add_argument(
        "--engine", choices=["tesseract", "easyocr"], default="tesseract",
        help="OCR backend (default: tesseract).",
    )
    parser.add_argument(
        "--interval", type=float, default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between captures (default: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_OUTPUT_DIR / f"transcript_{datetime.now():%Y%m%d_%H%M%S}.md",
        help="Output transcript (.txt or .md).",
    )
    parser.add_argument(
        "--tesseract-cmd", type=str, default=None,
        help="Full path to tesseract.exe if not on PATH (Windows).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )

    # --- Resolve bounding box ---
    if args.bbox:
        bbox = args.bbox
    else:
        logging.info("No --bbox provided. Launching interactive calibrator...")
        bbox = calibrate_bounding_box()
        logging.info(
            "Save this for next time:  --bbox %d,%d,%d,%d",
            bbox[0], bbox[1], bbox[2], bbox[3],
        )

    # --- OCR engine ---
    if args.engine == "tesseract":
        ocr: OCREngine = TesseractEngine(tesseract_cmd=args.tesseract_cmd)
    else:
        logging.info("Loading EasyOCR (first run downloads ~100MB model)...")
        ocr = EasyOCREngine()

    # --- Output file ---
    as_md = args.output.suffix.lower() == ".md"
    writer = TranscriptWriter(args.output, as_markdown=as_md)
    dedup = CaptionDeduplicator()

    try:
        capture_loop(bbox, ocr, writer, dedup, interval=args.interval)
    finally:
        writer.newline()
        logging.info("Transcript saved: %s", writer.path.resolve())


if __name__ == "__main__":
    main()