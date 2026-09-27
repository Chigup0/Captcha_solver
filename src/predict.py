"""Part 8 - predict the text of CAPTCHA image(s) with the learned templates.

    python src/predict.py path/to/captcha.jpg [more.jpg ...] [--method ncc] [--debug out_dir]

Output:
    Predicted: A7kP2
    Confidence/match score: mean 0.912, min 0.874 (margin to runner-up min 0.061)

The score is the matcher's similarity (NCC in [-1, 1] for `ncc`), not a probability;
the margin (best label - second-best label) is the more useful "am I sure" signal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config  # noqa: E402
from matcher import MATCHERS, TemplateSet, get_matcher  # noqa: E402
from preprocess import preprocess, stage_strip  # noqa: E402
from segment import get_segmenter  # noqa: E402


class Pipeline:
    def __init__(self, cfg: dict, method: str | None = None, tset: TemplateSet | None = None):
        self.cfg = cfg
        self.segmenter = get_segmenter(cfg)
        self.tset = tset or TemplateSet.load(cfg)
        self.matcher = get_matcher(method or cfg["matching"]["default_method"], self.tset, cfg)

    def predict(self, img: np.ndarray, keep_stages: bool = False) -> dict:
        crops = self.segmenter.split(img)
        preps = [preprocess(c.image, self.cfg, keep_stages) for c in crops]
        matches = self.matcher.predict(preps)
        return {
            "text": "".join(m.label for m in matches),
            "scores": [round(m.score, 4) for m in matches],
            "margins": [round(m.margin, 4) for m in matches],
            "angles": [None if np.isnan(m.angle) else round(m.angle, 1) for m in matches],
            "top3": [[(lab, round(s, 3)) for lab, s in m.top] for m in matches],
            "segmentation": crops[0].method,
            "_crops": crops, "_preps": preps, "_matches": matches,
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--config")
    ap.add_argument("--method", choices=list(MATCHERS))
    ap.add_argument("--debug", metavar="DIR", help="save preprocessing stages + matched template per char")
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pipe = Pipeline(cfg, args.method)

    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: cannot read image", file=sys.stderr)
            continue
        t0 = time.perf_counter()
        res = pipe.predict(img, keep_stages=bool(args.debug))
        ms = (time.perf_counter() - t0) * 1000
        public = {k: v for k, v in res.items() if not k.startswith("_")}
        if args.json:
            print(json.dumps({"image": path, **public, "ms": round(ms, 2)}))
        else:
            print(f"{path}")
            print(f"Predicted: {res['text']}")
            print(f"Confidence/match score: mean {np.mean(res['scores']):.3f}, min {min(res['scores']):.3f} "
                  f"(margin to runner-up min {min(res['margins']):.3f})")
            for i, (ch, s, mg, a, top) in enumerate(zip(res["text"], res["scores"], res["margins"],
                                                        res["angles"], res["top3"])):
                alt = ", ".join(f"{lab}:{sc:.3f}" for lab, sc in top[1:])
                print(f"  [{i}] {ch}  score {s:.3f}  margin {mg:.3f}  rot {a}  (next: {alt})")
            print(f"  {ms:.1f} ms, segmentation={res['segmentation']}")
        if args.debug:
            out = Path(args.debug)
            out.mkdir(parents=True, exist_ok=True)
            for i, (p, m) in enumerate(zip(res["_preps"], res["_matches"])):
                strip = stage_strip(p.stages, extra={f"match {m.label}": pipe.tset.images[m.template]})
                cv2.imwrite(str(out / f"{Path(path).stem}_{i}.png"), strip)


if __name__ == "__main__":
    main()
