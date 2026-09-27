"""Part 5 - compare a canonical character against the learned base-shape templates.

Candidate methods (pick one with predict.py --method):

  ncc         normalised cross-correlation against every template rotated over
              [-max_rot-margin, +max_rot+margin]; a (Q x F)(F x M) matrix product.
  chamfer     symmetric chamfer distance between edge maps, same rotation sweep.
  hu          log-scaled Hu moments (fully rotation invariant - so it cannot tell
              6/9, n/u, M/W apart; included as the classic baseline).
  ncc_deskew  experimental: deskew query and templates by principal axis, then a single
              NCC comparison (rotation normalisation instead of rotation search).
  ensemble    ncc - w * chamfer.

Every method returns a score per template (higher = better), optionally penalised by
the size mismatch |log(size_q / size_t)| (helps case pairs such as o/O, s/S, x/X).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import rpath  # noqa: E402
from preprocess import Prep, blur, deskew_moments, rotate_canonical  # noqa: E402


# --------------------------------------------------------------------------- shared helpers
def unit_rows(x: np.ndarray) -> np.ndarray:
    """Flatten, zero-mean and L2-normalise each row -> dot product == NCC."""
    x = x.reshape(len(x), -1).astype(np.float32)
    x = x - x.mean(1, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)


def rotation_angles(max_angle: float, step: float) -> np.ndarray:
    return np.arange(-max_angle, max_angle + 1e-6, step, dtype=np.float32)


def rotated_stack(canons: np.ndarray, angles: np.ndarray, margin: int) -> np.ndarray:
    """(N, S, S) -> (N, A, S, S): every canonical image rotated by every angle."""
    return np.stack([np.stack([rotate_canonical(c, float(a), margin=margin) for a in angles]) for c in canons])


def hu_features(canon: np.ndarray) -> np.ndarray:
    hu = cv2.HuMoments(cv2.moments((canon > 0.5).astype(np.uint8), binaryImage=True)).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)


def edges(binary: np.ndarray) -> np.ndarray:
    b = binary.astype(np.uint8)
    return (b - cv2.erode(b, np.ones((3, 3), np.uint8))) > 0


def dist_transform(edge: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)


# --------------------------------------------------------------------------- template set
@dataclass
class TemplateSet:
    images: np.ndarray        # (N, S, S) float32 upright canonical templates
    labels: np.ndarray        # (N,) str
    shape_ids: np.ndarray     # (N,) int
    sizes: np.ndarray         # (N,) float, median radius of gyration of the members

    @classmethod
    def load(cls, cfg) -> "TemplateSet":
        path = rpath(cfg["paths"]["templates_dir"]) / "templates.npz"
        if not path.exists():
            raise SystemExit(f"{path} not found - run src/build_templates.py first.")
        z = np.load(path, allow_pickle=False)
        return cls(z["images"], z["labels"].astype(str), z["shape_ids"], z["sizes"])

    def save(self, cfg) -> Path:
        path = rpath(cfg["paths"]["templates_dir"]) / "templates.npz"
        np.savez_compressed(path, images=self.images, labels=self.labels.astype("U1"),
                            shape_ids=self.shape_ids, sizes=self.sizes)
        return path


def load_matcher_params(cfg) -> dict:
    p = {"size_penalty": cfg["matching"]["size_penalty"],
         "ensemble_chamfer_weight": cfg["matching"]["ensemble_chamfer_weight"]}
    path = rpath(cfg["paths"]["templates_dir"]) / "matcher_params.json"
    if path.exists():
        p.update(json.loads(path.read_text()))
    return p


# --------------------------------------------------------------------------- matchers
@dataclass
class Match:
    label: str
    score: float           # best template score for the winning label
    margin: float          # winner - runner-up (different label)
    template: int          # index into TemplateSet
    angle: float           # estimated glyph rotation (deg, CCW), nan if not estimated
    top: list              # [(label, score), ...] top-3 labels


class Matcher:
    name = "base"

    def __init__(self, tset: TemplateSet, cfg: dict, params: dict | None = None):
        self.t = tset
        self.cfg = cfg
        self.params = params or load_matcher_params(cfg)
        m = cfg["matching"]
        self.margin_px = cfg["preprocess"]["margin"]
        self.sigma = cfg["preprocess"]["blur_sigma"]
        self.angles = rotation_angles(m["max_rotation"] + m["rotation_margin"], m["rotation_step"])
        self.uniq = np.unique(self.t.labels)
        self.label_idx = np.searchsorted(self.uniq, self.t.labels)

    # subclasses: (Q, N) template scores and (Q, N) best angle
    def template_scores(self, preps: list[Prep]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def scores(self, preps):
        s, ang = self.template_scores(preps)
        lam = self.params.get("size_penalty", 0.0)
        if lam:
            q = np.array([max(p.size, 1e-3) for p in preps])[:, None]
            s = s - lam * np.abs(np.log(q / np.maximum(self.t.sizes[None], 1e-3)))
        return s, ang

    def predict(self, preps: list[Prep]) -> list[Match]:
        s, ang = self.scores(preps)
        out = []
        for i in range(len(preps)):
            per_label = np.full(len(self.uniq), -np.inf)
            np.maximum.at(per_label, self.label_idx, s[i])
            order = np.argsort(-per_label)
            best_t = int(np.argmax(np.where(self.label_idx == order[0], s[i], -np.inf)))
            second = per_label[order[1]] if len(order) > 1 else -np.inf
            out.append(Match(self.uniq[order[0]], float(per_label[order[0]]),
                             float(per_label[order[0]] - second), best_t, float(ang[i, best_t]),
                             [(self.uniq[j], float(per_label[j])) for j in order[:3]]))
        return out


class NCCMatcher(Matcher):
    name = "ncc"

    def __init__(self, tset, cfg, params=None):
        super().__init__(tset, cfg, params)
        rot = rotated_stack(tset.images, self.angles, self.margin_px)              # (N, A, S, S)
        self.N, self.A = rot.shape[:2]
        self.bank = unit_rows(np.stack([blur(r, self.sigma) for r in rot.reshape(-1, *rot.shape[2:])]))

    def template_scores(self, preps):
        q = unit_rows(np.stack([blur(p.canonical, self.sigma) for p in preps]))
        s = (q @ self.bank.T).reshape(len(preps), self.N, self.A)
        return s.max(2), self.angles[s.argmax(2)]


class ChamferMatcher(Matcher):
    name = "chamfer"

    def __init__(self, tset, cfg, params=None):
        super().__init__(tset, cfg, params)
        rot = rotated_stack(tset.images, self.angles, self.margin_px)
        self.N, self.A = rot.shape[:2]
        flat = rot.reshape(-1, *rot.shape[2:]) > 0.5
        e = np.stack([edges(b) for b in flat])
        self.edge_bank = e.reshape(len(e), -1).astype(np.float32)                  # (M, F)
        self.edge_count = np.maximum(self.edge_bank.sum(1), 1)
        self.dt_bank = np.stack([dist_transform(x) for x in e]).reshape(len(e), -1)  # (M, F)

    def cost(self, canon):
        e = edges(canon > 0.5)
        idx = np.flatnonzero(e)
        if len(idx) == 0:
            return np.full(len(self.dt_bank), 1e3, np.float32)
        q_to_t = self.dt_bank[:, idx].mean(1)
        t_to_q = (self.edge_bank @ dist_transform(e).ravel()) / self.edge_count
        return 0.5 * (q_to_t + t_to_q)

    def template_scores(self, preps):
        c = np.stack([self.cost(p.canonical) for p in preps]).reshape(len(preps), self.N, self.A)
        return -c.min(2), self.angles[c.argmin(2)]


class HuMatcher(Matcher):
    name = "hu"

    def __init__(self, tset, cfg, params=None):
        super().__init__(tset, cfg, params)
        self.feats = np.stack([hu_features(t) for t in tset.images])[:, :6]  # 7th flips sign on mirroring

    def template_scores(self, preps):
        q = np.stack([hu_features(p.canonical) for p in preps])[:, :6]
        d = np.abs(q[:, None, :] - self.feats[None]).sum(-1)
        return -d, np.full(d.shape, np.nan, np.float32)


class DeskewNCCMatcher(Matcher):
    name = "ncc_deskew"

    def __init__(self, tset, cfg, params=None):
        super().__init__(tset, cfg, params)
        self.bank = unit_rows(np.stack([blur(deskew_moments(t, self.margin_px)[0], self.sigma)
                                        for t in tset.images]))

    def template_scores(self, preps):
        q = unit_rows(np.stack([blur(deskew_moments(p.canonical, self.margin_px)[0], self.sigma)
                                for p in preps]))
        s = q @ self.bank.T
        return s, np.full(s.shape, np.nan, np.float32)


class EnsembleMatcher(Matcher):
    name = "ensemble"

    def __init__(self, tset, cfg, params=None):
        super().__init__(tset, cfg, params)
        self.ncc = NCCMatcher(tset, cfg, params)
        self.chamfer = ChamferMatcher(tset, cfg, params)

    def template_scores(self, preps):
        s1, a1 = self.ncc.template_scores(preps)
        s2, _ = self.chamfer.template_scores(preps)
        return s1 + self.params.get("ensemble_chamfer_weight", 0.05) * s2, a1


MATCHERS = {c.name: c for c in (NCCMatcher, ChamferMatcher, HuMatcher, DeskewNCCMatcher, EnsembleMatcher)}


def get_matcher(name: str, tset: TemplateSet, cfg: dict, params: dict | None = None) -> Matcher:
    return MATCHERS[name](tset, cfg, params)
