# Evaluation against supplied transcription

AI-assisted candidate transcription; not independently verified ground truth.

All 114 pages parsed. 15 pages flagged in the supplied uncertainty notes; six inline uncertainty markers. The original file is unchanged.

Full-book scores use the supplied best readings in uncertainty markers. The second analysis excludes every flagged page for every engine; this is a sensitivity check, not a verified subset.

## 114-page provisional comparison (114 pages)

| Engine | WER | CER | Word edits | Reference words |
|---|---:|---:|---:|---:|
| indra | 0.385% | 0.233% | 233 | 60543 |
| tesseract | 1.069% | 0.386% | 647 | 60543 |
| easyocr | 16.895% | 7.676% | 10229 | 60543 |
| paddleocr_tiny | 4.683% | 1.074% | 2835 | 60543 |

Paired WER differences (percentage points; negative favors left). Descriptive 95% page-bootstrap intervals within this book:

- indra vs tesseract: -0.684 [-0.781, -0.595]
- indra vs easyocr: -16.511 [-17.283, -15.798]
- indra vs paddleocr_tiny: -4.298 [-4.877, -3.783]
- tesseract vs easyocr: -15.827 [-16.572, -15.117]
- tesseract vs paddleocr_tiny: -3.614 [-4.174, -3.118]
- easyocr vs paddleocr_tiny: 12.213 [11.726, 12.711]

## Excluding all flagged pages (99 pages)

| Engine | WER | CER | Word edits | Reference words |
|---|---:|---:|---:|---:|
| indra | 0.225% | 0.038% | 118 | 52470 |
| tesseract | 0.888% | 0.133% | 466 | 52470 |
| easyocr | 16.749% | 7.425% | 8788 | 52470 |
| paddleocr_tiny | 4.610% | 0.861% | 2419 | 52470 |

Paired WER differences (percentage points; negative favors left). Descriptive 95% page-bootstrap intervals within this book:

- indra vs tesseract: -0.663 [-0.759, -0.576]
- indra vs easyocr: -16.524 [-17.278, -15.811]
- indra vs paddleocr_tiny: -4.385 [-5.010, -3.805]
- tesseract vs easyocr: -15.860 [-16.597, -15.169]
- tesseract vs paddleocr_tiny: -3.722 [-4.314, -3.170]
- easyocr vs paddleocr_tiny: 12.138 [11.670, 12.618]

## Quote-style sensitivity

Additionally map U+2018/U+2019 to ASCII apostrophe and U+201C/U+201D to ASCII double quote, on reference and prediction; preserve case and other punctuation. Supplemental analysis only, not a replacement for the primary strict scores.

| Engine | 114-page WER | 114-page CER | 99-page WER | 99-page CER |
|---|---:|---:|---:|---:|
| indra | 0.345% | 0.222% | 0.179% | 0.025% |
| tesseract | 0.910% | 0.354% | 0.724% | 0.099% |
| easyocr | 15.567% | 7.320% | 15.388% | 7.069% |
| paddleocr_tiny | 2.568% | 0.536% | 2.499% | 0.318% |

Candidate vs embedded PDF text: 2.002% WER, 0.404% CER; 3/114 exact normalized pages. Agreement cannot establish independent provenance or correctness.

Flagged pages: 42, 43, 52, 56, 57, 60, 61, 62, 65, 70, 78, 80, 93, 95, 96.

Spot-checked saved page images 2, 52 and 76; no complete independent visual verification. Page 52 contains visibly corrupted amount glyphs. On page 76 the candidate uses straight apostrophes in printed HTML attributes where the image shows curly apostrophes, demonstrating that even unflagged pages need review. Original transcription left unchanged. Transcriber model/version and generation records were not supplied.

No OCR rerun. Speed/RAM remain the original four-engine experiment, not new measurements.

The Medium draft retains the original reference scores. This supplemental candidate-reference analysis must not be presented as verified accuracy.

The word-level book-text comparison remains local and is excluded from Git.

Transcription SHA256: `96c5712d9e334c62f00e4e56a0db262cf078ea05fb18a1d77e0398be7121676b`
