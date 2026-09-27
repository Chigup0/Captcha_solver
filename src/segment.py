"""Part 2 - split each CAPTCHA into its characters.

Segmenters share one interface (`Segmenter.split(img) -> list[CharCrop]`), so the method
can be swapped via config `segmentation.method` without touching callers.

  equal_width   - N equal cells over the horizontal ink extent, with overlap; each cell
                  keeps the connected components whose centroid lies inside the cell.
  color_kmeans  - each glyph is drawn in its own colour, so ink pixels are clustered on
                  (Lab colour, x). Separates touching glyphs that a column split cannot.
                  Falls back to equal_width when the clusters look implausible.
  components    - (default) every glyph is one blob of one colour: take connected components,
                  merge the most similar pair (colour + horizontal gap) while there are more
                  than N, split the widest group by colour/x while there are fewer.

    python src/segment.py                 # every image in metadata.csv -> data/chars/
    python src/segment.py --debug 30      # also save 30 overlay images
"""
from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import image_path, load_config, load_metadata, log, rpath, setup_logging  # noqa: E402
from preprocess import background_color, ink_mask, ink_strength, remove_small  # noqa: E402


@dataclass
class CharCrop:
    pos: int
    image: np.ndarray        # BGR, glyph isolated on the background colour
    mask: np.ndarray         # uint8 0/255, same size as image
    bbox: tuple              # x, y, w, h in the source image
    method: str


class Segmenter(ABC):
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seg = cfg["segmentation"]
        self.n = cfg["captcha"]["n_chars"]

    def ink(self, img):
        bg = background_color(img)
        mask = ink_mask(ink_strength(img, bg), self.seg["min_strength"])
        return bg, remove_small(mask, self.seg["min_component_area"])

    def make_crops(self, img, bg, label_map: np.ndarray, method: str) -> list[CharCrop]:
        """label_map: int array, pixel -> char position (-1 = background)."""
        pad = self.seg["crop_pad"]
        H, W = img.shape[:2]
        crops = []
        for pos in range(self.n):
            m = (label_map == pos).astype(np.uint8) * 255
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                crops.append(CharCrop(pos, np.full((8, 8, 3), bg, np.uint8), np.zeros((8, 8), np.uint8),
                                      (0, 0, 0, 0), method + ":empty"))
                continue
            x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)
            y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
            sub = img[y0:y1, x0:x1].copy()
            sm = m[y0:y1, x0:x1]
            sub[sm == 0] = bg.astype(np.uint8)
            crops.append(CharCrop(pos, sub, sm, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), method))
        return crops

    @abstractmethod
    def split(self, img: np.ndarray) -> list[CharCrop]: ...


class EqualWidthSegmenter(Segmenter):
    name = "equal_width"

    def assign(self, mask: np.ndarray) -> np.ndarray:
        cols = np.nonzero(mask.any(axis=0))[0]
        label_map = np.full(mask.shape, -1, np.int32)
        if len(cols) == 0:
            return label_map
        x0, x1 = cols.min(), cols.max() + 1
        cell = (x1 - x0) / self.n
        n, lab, stats, cent = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        for c in range(1, n):
            pos = int(np.clip((cent[c, 0] - x0) // cell, 0, self.n - 1))
            comp = lab == c
            # a wide component (touching glyphs) is cut at the cell borders, widened by the overlap
            if stats[c, cv2.CC_STAT_WIDTH] > 1.3 * cell:
                xs = np.arange(mask.shape[1])
                cell_of_x = np.clip((xs - x0) // cell, 0, self.n - 1).astype(np.int32)
                label_map[comp] = np.broadcast_to(cell_of_x, mask.shape)[comp]
            else:
                label_map[comp] = pos
        return label_map

    def split(self, img):
        bg, mask = self.ink(img)
        return self.make_crops(img, bg, self.assign(mask), self.name)


class ColorKMeansSegmenter(Segmenter):
    name = "color_kmeans"

    def split(self, img):
        bg, mask = self.ink(img)
        ys, xs = np.nonzero(mask)
        fallback = EqualWidthSegmenter(self.cfg)
        if len(xs) < self.n * 10:
            return fallback.make_crops(img, bg, fallback.assign(mask), "equal_width(fallback)")
        k = self.seg["color_kmeans"]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[ys, xs]
        feats = np.column_stack([
            (lab[:, 1] - 128) * k["w_color"], (lab[:, 2] - 128) * k["w_color"],
            lab[:, 0] * (100 / 255) * k["w_lightness"], xs * k["w_x"],
        ]).astype(np.float32)
        # initialise from the equal-width split: deterministic and position-ordered
        init = fallback.assign(mask)[ys, xs]
        centers = np.stack([feats[init == p].mean(0) if (init == p).any() else feats.mean(0)
                            for p in range(self.n)])
        assign = init.copy()
        for _ in range(20):
            d = ((feats[:, None, :] - centers[None]) ** 2).sum(-1)
            new = d.argmin(1)
            if (new == assign).all():
                break
            assign = new
            for p in range(self.n):
                if (assign == p).any():
                    centers[p] = feats[assign == p].mean(0)
        # order clusters left -> right
        med_x = [np.median(xs[assign == p]) if (assign == p).any() else 1e9 for p in range(self.n)]
        remap = np.empty(self.n, np.int32)
        remap[np.argsort(med_x)] = np.arange(self.n)
        assign = remap[assign]
        # plausibility check
        text_w = xs.max() - xs.min() + 1
        for p in range(self.n):
            sel = assign == p
            if sel.sum() < k["min_cluster_frac"] * len(xs) or \
                    np.percentile(xs[sel], 98) - np.percentile(xs[sel], 2) > k["max_cluster_span"] * text_w:
                return fallback.make_crops(img, bg, fallback.assign(mask), "equal_width(fallback)")
        label_map = np.full(mask.shape, -1, np.int32)
        label_map[ys, xs] = assign
        # drop stray fragments (e.g. anti-aliased pixels grabbed by a neighbour's colour)
        for p in range(self.n):
            m = (label_map == p).astype(np.uint8) * 255
            keep = remove_small(m, self.seg["min_component_area"]) > 0
            label_map[(label_map == p) & ~keep] = -1
        return self.make_crops(img, bg, label_map, self.name)


class ComponentSegmenter(Segmenter):
    name = "components"

    def split(self, img):
        bg, mask = self.ink(img)
        n_cc, cc = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)[:2]
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        groups = [np.nonzero(cc == c) for c in range(1, n_cc)]  # each: (ys, xs)
        if len(groups) == 0:
            fallback = EqualWidthSegmenter(self.cfg)
            return fallback.make_crops(img, bg, fallback.assign(mask), "equal_width(fallback)")
        opts = self.seg.get("components", {})
        w_gap, speck_frac = opts.get("w_gap", 2.0), opts.get("speck_frac", 0.15)

        def color(g):
            return np.median(lab[g[0], g[1]], axis=0) * np.array([0.5, 1, 1], np.float32)  # L* counts half

        def pixel_gap(a, b):  # distance between the closest pixels of two blobs
            pa, pb = np.column_stack(a).astype(np.float32), np.column_stack(b).astype(np.float32)
            return float(np.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1).min()))

        # too many pieces: merge until there are n
        while len(groups) > self.n:
            sizes = [len(g[1]) for g in groups]
            s = int(np.argmin(sizes))
            if sizes[s] < speck_frac * np.median(sizes):
                # a speck (i/j dot) is never a glyph: attach it to the similar-coloured blob with the
                # closest pixel - bounding boxes mislead when the i/j is rotated and its dot sits sideways
                j = min((j for j in range(len(groups)) if j != s),
                        key=lambda j: np.linalg.norm(color(groups[s]) - color(groups[j]))
                        + w_gap * pixel_gap(groups[s], groups[j]))
                i, j = min(s, j), max(s, j)
            else:  # broken strokes / leftovers: merge the most similar pair (colour + horizontal gap)
                best, pair = np.inf, None
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        (yi, xi), (yj, xj) = groups[i], groups[j]
                        gap = max(0, max(xi.min(), xj.min()) - min(xi.max(), xj.max()))
                        cost = np.linalg.norm(color(groups[i]) - color(groups[j])) + w_gap * gap
                        if cost < best:
                            best, pair = cost, (i, j)
                i, j = pair
            groups[i] = (np.concatenate([groups[i][0], groups[j][0]]), np.concatenate([groups[i][1], groups[j][1]]))
            del groups[j]

        # too few (touching glyphs): split the widest group in two by colour + x
        while len(groups) < self.n:
            i = max(range(len(groups)), key=lambda g: np.ptp(groups[g][1]))
            ys, xs = groups[i]
            if len(xs) < 4:
                break
            f = np.column_stack([lab[ys, xs] * np.array([0.5, 1, 1], np.float32), xs * 1.5]).astype(np.float32)
            a = (xs > np.median(xs)).astype(np.int32)  # init: left / right half
            for _ in range(20):
                if a.min() == a.max():
                    break
                c = np.stack([f[a == k].mean(0) for k in (0, 1)])
                new = ((f[:, None] - c[None]) ** 2).sum(-1).argmin(1)
                if (new == a).all():
                    break
                a = new
            if a.min() == a.max():
                break
            groups[i:i + 1] = [(ys[a == 0], xs[a == 0]), (ys[a == 1], xs[a == 1])]

        groups.sort(key=lambda g: np.median(g[1]))  # left -> right
        label_map = np.full(mask.shape, -1, np.int32)
        for p, (ys, xs) in enumerate(groups[: self.n]):
            label_map[ys, xs] = p
        return self.make_crops(img, bg, label_map, self.name)


SEGMENTERS = {c.name: c for c in (EqualWidthSegmenter, ColorKMeansSegmenter, ComponentSegmenter)}


def get_segmenter(cfg: dict, method: str | None = None) -> Segmenter:
    return SEGMENTERS[method or cfg["segmentation"]["method"]](cfg)


def overlay(img: np.ndarray, crops: list[CharCrop]) -> np.ndarray:
    vis = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    for c in crops:
        x, y, w, h = [v * 3 for v in c.bbox]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(vis, str(c.pos), (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    strip = np.hstack([cv2.copyMakeBorder(cv2.resize(c.image, (60, 60)), 2, 2, 2, 2,
                                          cv2.BORDER_CONSTANT, value=(128, 128, 128)) for c in crops])
    strip = cv2.copyMakeBorder(strip, 0, 0, 0, max(0, vis.shape[1] - strip.shape[1]),
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return np.vstack([vis, strip[:, :vis.shape[1]]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--method", choices=list(SEGMENTERS))
    ap.add_argument("--debug", type=int, default=20, help="number of overlay images to save")
    ap.add_argument("--image", help="segment just this image and save an overlay")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    seg = get_segmenter(cfg, args.method)
    dbg = rpath(cfg["paths"]["debug_dir"]) / "segmentation"
    dbg.mkdir(parents=True, exist_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        crops = seg.split(img)
        out = dbg / f"{Path(args.image).stem}_{seg.name}.png"
        cv2.imwrite(str(out), overlay(img, crops))
        print(f"{[c.method for c in crops][0]} -> {out}")
        return

    meta = load_metadata(cfg)
    crop_dir = rpath(cfg["paths"]["chars_dir"]) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows, n_fallback = [], 0
    for i, r in enumerate(meta.itertuples(index=False)):
        img = cv2.imread(str(image_path(cfg, r.filename)))
        if img is None:
            log.warning("cannot read %s", r.filename)
            continue
        crops = seg.split(img)
        n_fallback += "fallback" in crops[0].method
        if i < args.debug:
            cv2.imwrite(str(dbg / f"{Path(r.filename).stem}.png"), overlay(img, crops))
        for c in crops:
            name = f"{r.captcha_id}_{int(r.request_no):03d}_{c.pos}.png"
            cv2.imwrite(str(crop_dir / name), c.image)
            rows.append(dict(crop_path=str(Path(cfg["paths"]["chars_dir"]) / "crops" / name),
                             captcha_id=r.captcha_id, request_no=int(r.request_no), pos=c.pos,
                             source_image=r.filename, seg_method=c.method,
                             bbox_x=c.bbox[0], bbox_y=c.bbox[1], bbox_w=c.bbox[2], bbox_h=c.bbox[3]))
        if (i + 1) % 500 == 0:
            log.info("%d/%d images", i + 1, len(meta))
    idx = pd.DataFrame(rows)
    idx.to_csv(rpath(cfg["paths"]["chars_index"]), index=False)
    log.info("%d crops from %d images (%d used the equal-width fallback) -> %s",
             len(idx), len(meta), n_fallback, cfg["paths"]["chars_index"])


if __name__ == "__main__":
    main()
