"""Batch evaluation.

Put audio files, reference transcripts and reference summaries in a folder:
    data/clip1.wav  data/clip1.txt (reference transcript)  data/clip1.summary.txt (reference summary)
Run:  python evaluate.py data --language Malayalam --out results.csv
"""
import argparse
import csv
from pathlib import Path

from speech2summary import metrics, speech_to_summary

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--language", required=True)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    rows = []
    for audio in sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in AUDIO_EXT):
        ref_t, ref_s = audio.with_suffix(".txt"), audio.with_suffix(".summary.txt")
        res = speech_to_summary(str(audio), args.language)
        row = {"file": audio.name, "asr_s": round(res.asr_seconds, 2), "summary_s": round(res.summary_seconds, 2)}
        if ref_t.exists():
            ref = ref_t.read_text(encoding="utf-8").strip()
            row["WER"] = round(metrics.wer(ref, res.transcript), 4)
            row["CER"] = round(metrics.cer(ref, res.transcript), 4)
        if ref_s.exists():
            r = metrics.rouge(ref_s.read_text(encoding="utf-8").strip(), res.summary)
            row.update({k: round(v, 4) for k, v in r.items()})
        rows.append(row)
        print(row)

    if not rows:
        raise SystemExit("no audio files found")
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for c in ("WER", "CER", "rouge1", "rouge2", "rougeL", "asr_s", "summary_s"):
        vals = [r[c] for r in rows if c in r]
        if vals:
            print(f"mean {c}: {sum(vals) / len(vals):.4f}  (n={len(vals)})")


if __name__ == "__main__":
    main()
