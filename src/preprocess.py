"""Part 4 - colour/rotation/scale normalisation of a single character crop.

Stages (all kept when keep_stages=True, for failure inspection):
    original -> gray -> ink (colour-invariant foreground strength) -> binary mask
    -> cleaned mask -> bbox crop -> moments-deskewed (experimental) -> canonical

The canonical form is a float SxS image in [0, 1] (1 = ink): tight bounding box,
padded to a square, resized. Colour is removed by measuring *distance from the
background colour* rather than any fixed RGB value; scale and translation are removed
by the bbox + resize; rotation is handled either by rotating templates at match time
(matcher.py, default) or by the experimental moments deskew stage.

    python src/preprocess.py --image some_crop.png          # dump a stage strip
    python src/preprocess.py --from-index 50                # strips for 50 random crops
"""
from __future__ import annotations

import argparse
import colorsys
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, rpath  # noqa: E402


# --------------------------------------------------------------------------- foreground
def background_color(img: np.ndarray, border: int = 2) -> np.ndarray:
    b = np.concatenate([img[:border].reshape(-1, 3), img[-border:].reshape(-1, 3),
                        img[:, :border].reshape(-1, 3), img[:, -border:].reshape(-1, 3)])
    return np.median(b, axis=0).astype(np.float32)


def ink_strength(img: np.ndarray, bg: np.ndarray | None = None) -> np.ndarray:
    """Euclidean RGB distance from the background colour, scaled to [0, 1].

    Works for any glyph colour (dark, light, saturated) on a roughly uniform background,
    unlike plain grayscale, where e.g. yellow on white nearly vanishes.
    """
    if bg is None:
        bg = background_color(img)
    d = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    return np.clip(d / 441.673, 0, 1)


def ink_mask(strength: np.ndarray, min_strength: float) -> np.ndarray:
    s8 = (strength * 255).astype(np.uint8)
    thr, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr, min_strength * 255)
    return (s8 > thr).astype(np.uint8) * 255


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    if n > 1 and not keep.any():  # never delete everything: keep the largest component
        keep[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])] = True
    return (keep[lab]).astype(np.uint8) * 255


# --------------------------------------------------------------------------- geometry
def canonicalize(img: np.ndarray, size: int, margin: int = 2, thresh: float = 0.25) -> np.ndarray:
    """Tight bbox of ink (>thresh), pad to square (centred), resize to size x size, float [0,1]."""
    f = img.astype(np.float32)
    if f.max() > 1.0:
        f /= 255.0
    ys, xs = np.nonzero(f > thresh)
    out = np.zeros((size, size), np.float32)
    if len(xs) == 0:
        return out
    f = f[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = f.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = f
    inner = size - 2 * margin
    interp = cv2.INTER_AREA if side > inner else cv2.INTER_LINEAR
    out[margin:margin + inner, margin:margin + inner] = cv2.resize(sq, (inner, inner), interpolation=interp)
    return out


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a float image counter-clockwise by `angle` degrees on an enlarged canvas."""
    h, w = img.shape
    side = int(np.ceil(np.hypot(h, w))) + 2
    canvas = np.zeros((side, side), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    m = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    return cv2.warpAffine(canvas, m, (side, side), flags=cv2.INTER_LINEAR, borderValue=0)


def rotate_canonical(canon: np.ndarray, angle: float, size: int | None = None, margin: int = 2) -> np.ndarray:
    size = size or canon.shape[0]
    if angle == 0:
        return canon.copy()
    return canonicalize(rotate(canon, angle), size, margin)


def moments_angle(mask: np.ndarray) -> float:
    """Principal-axis deviation from the nearest image axis, in (-45, 45] degrees.

    Experimental: symmetric glyphs (O, 0, X...) have no stable axis, and the snap to the
    nearest axis flips near +-45. Kept as a stage + benchmarked method, not the default.
    """
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return 0.0
    theta = 0.5 * np.degrees(np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"]))  # axis vs x, y down
    dev = theta - 90 * np.round(theta / 90)
    return float(dev)


def deskew_moments(canon: np.ndarray, margin: int = 2) -> tuple[np.ndarray, float]:
    a = moments_angle(canon > 0.5)
    # image y axis points down, so an axis at +a (clockwise-looking) is undone by rotating +a CCW
    return rotate_canonical(canon, a, margin=margin), a


def radius_of_gyration(mask: np.ndarray) -> float:
    """Rotation-invariant size (px) - used to separate case pairs such as o/O, s/S."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    return float(np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean()))


# --------------------------------------------------------------------------- colour naming
HUE_BINS = [(15, "red"), (40, "orange"), (70, "yellow"), (160, "green"), (200, "cyan"),
            (260, "blue"), (300, "purple"), (340, "magenta"), (360, "red")]


def color_name(bgr) -> str:
    b, g, r = [float(c) / 255 for c in bgr]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.25:
        return "black"
    if s < 0.2:
        return "gray"
    hue = h * 360
    for upper, name in HUE_BINS:
        if hue < upper:
            return name
    return "red"


# --------------------------------------------------------------------------- pipeline
@dataclass
class Prep:
    canonical: np.ndarray            # float32 SxS, 1 = ink
    size: float                      # radius of gyration, source pixels
    color_bgr: tuple                 # median ink colour
    ok: bool = True
    stages: dict = field(default_factory=dict)


def preprocess(crop: np.ndarray, cfg: dict, keep_stages: bool = False) -> Prep:
    p = cfg["preprocess"]
    size, margin = p["canonical_size"], p["margin"]
    stages = {}
    if keep_stages:
        stages["original"] = crop.copy()
        stages["gray"] = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    strength = ink_strength(crop)
    mask = ink_mask(strength, cfg["segmentation"]["min_strength"])
    clean = remove_small(mask, p["min_component_area"])
    ys, xs = np.nonzero(clean)
    if len(xs) == 0:
        return Prep(np.zeros((size, size), np.float32), 0.0, (0, 0, 0), ok=False, stages=stages)
    color = tuple(int(c) for c in np.median(crop[ys, xs], axis=0))
    canon = canonicalize(clean, size, margin)
    if keep_stages:
        stages["ink"] = (strength * 255).astype(np.uint8)
        stages["mask"] = mask
        stages["clean"] = clean
        stages["bbox"] = clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        stages["canonical"] = (canon * 255).astype(np.uint8)
        stages["deskew_moments"] = (deskew_moments(canon, margin)[0] * 255).astype(np.uint8)
    return Prep(canon, radius_of_gyration(clean), color, True, stages)


def blur(canon: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(canon, (0, 0), sigma) if sigma > 0 else canon


def stage_strip(stages: dict, height: int = 80, extra: dict | None = None) -> np.ndarray:
    """Concatenate all stages side by side, labelled, for visual debugging."""
    tiles = []
    for name, im in list(stages.items()) + list((extra or {}).items()):
        if im.dtype != np.uint8:
            im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        h, w = im.shape[:2]
        im = cv2.resize(im, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_NEAREST)
        cap = np.full((16, im.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, name[:14], (1, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
        tile = np.vstack([cap, im])
        tiles.append(cv2.copyMakeBorder(tile, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(200, 200, 200)))
    return np.hstack(tiles)


def canon_cache(cfg: dict, idx, log_every: int = 5000) -> dict:
    """Preprocess every crop in the index once; cached in data/chars/canon.npz.

    Returns dict(canon (n,S,S) float16, size (n,), color (n,3) uint8, ok (n,)), aligned
    to idx.crop_path. The cache is rebuilt when the crop list or preprocess config changes.
    """
    import hashlib
    import json
    key = hashlib.sha1((json.dumps(cfg["preprocess"], sort_keys=True) +
                        json.dumps(cfg["segmentation"]["min_strength"]) +
                        "\n".join(idx["crop_path"])).encode()).hexdigest()
    path = rpath(cfg["paths"]["chars_dir"]) / "canon.npz"
    if path.exists():
        z = np.load(path)
        if str(z["key"]) == key:
            return {k: z[k] for k in ("canon", "size", "color", "ok")}
    canon, size, color, ok = [], [], [], []
    for i, p in enumerate(idx["crop_path"]):
        pr = preprocess(cv2.imread(str(rpath(p))), cfg)
        canon.append(pr.canonical.astype(np.float16))
        size.append(pr.size)
        color.append(pr.color_bgr)
        ok.append(pr.ok)
        if log_every and (i + 1) % log_every == 0:
            print(f"  preprocessed {i + 1}/{len(idx)}", flush=True)
    out = dict(canon=np.stack(canon), size=np.array(size, np.float32),
               color=np.array(color, np.uint8), ok=np.array(ok))
    np.savez_compressed(path, key=key, **out)
    return out


def canon_for(cfg: dict, crop_paths) -> dict:
    """Cached preprocessing results for a subset of crops (looked up in the full-index cache)."""
    import pandas as pd
    idx = pd.read_csv(rpath(cfg["paths"]["chars_index"]), dtype={"captcha_id": str}, keep_default_na=False)
    cache = canon_cache(cfg, idx)
    row = pd.Series(np.arange(len(idx)), index=idx["crop_path"])[list(crop_paths)].to_numpy()
    return {k: v[row] for k, v in cache.items()}


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--image", help="a single character crop")
    ap.add_argument("--from-index", type=int, metavar="N", help="N random crops from the chars index")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = rpath(cfg["paths"]["debug_dir"]) / "preprocess"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if args.image:
        paths = [Path(args.image)]
    elif args.from_index:
        idx = pd.read_csv(rpath(cfg["paths"]["chars_index"]), dtype=str, keep_default_na=False)
        paths = [rpath(p) for p in idx.sample(min(args.from_index, len(idx)), random_state=0)["crop_path"]]
    for p in paths:
        prep = preprocess(cv2.imread(str(p)), cfg, keep_stages=True)
        cv2.imwrite(str(out / f"{p.stem}_stages.png"), stage_strip(prep.stages))
    print(f"wrote {len(paths)} stage strips to {out}")


if __name__ == "__main__":
    main()
