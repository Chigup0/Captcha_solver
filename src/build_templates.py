"""Part 6 - discover the base shapes of every character from labelled TRAIN crops only.

For each character:
  1. rotation-aligned distance between every pair of samples
       D_ij = 1 - max_theta NCC(rotate(x_i, theta), x_j),  theta in +-align_range
  2. agglomerative clustering (average linkage) into k=2; the split is accepted only if
     silhouette >= min_silhouette and the minority cluster >= min_minority_frac.
     Otherwise the character is REPORTED as "one_shape_or_unseparable" and gets a single
     template - never two invented ones.
  3. per cluster: medoid, outlier rejection, rotate every member to upright
     (alignment angle to the medoid minus its median, assuming the generator's rotations
     are centred on 0), average -> template.

Outputs: templates/<upper_A|lower_a|digit_7>/shape_{1,2}.png, templates/templates.npz,
templates/report.{json,md}, templates/_review/<char>.png (members per cluster).

    python src/build_templates.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from common import label_dir, load_config, log, rpath, setup_logging  # noqa: E402
from matcher import TemplateSet, rotated_stack, rotation_angles, unit_rows  # noqa: E402
from preprocess import blur, canon_for, canonicalize, rotate_canonical  # noqa: E402


def pairwise(X: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """S[i, j] = best NCC of rotated x_i vs x_j, and the angle achieving it."""
    t = cfg["templates"]
    sigma, margin = cfg["preprocess"]["blur_sigma"], cfg["preprocess"]["margin"]
    angles = rotation_angles(t["align_range"], t["align_step"])
    rot = rotated_stack(X, angles, margin)                                  # (n, A, S, S)
    n, A = rot.shape[:2]
    V = unit_rows(np.stack([blur(r, sigma) for r in rot.reshape(-1, *rot.shape[2:])]))
    U = unit_rows(np.stack([blur(x, sigma) for x in X]))
    sim = (V @ U.T).reshape(n, A, n)
    return sim.max(1), angles[sim.argmax(1)]


def make_template(X, members, S, ang, margin):
    """Medoid -> outlier rejection -> upright alignment -> mean."""
    D = 1 - S[np.ix_(members, members)]
    m_local = int(np.argmin(D.sum(1)))
    medoid = members[m_local]
    d = D[:, m_local]
    mad = np.median(np.abs(d - np.median(d))) + 1e-6
    inl = members[(d <= min(0.6, np.median(d) + 3 * 1.4826 * mad))]
    phi = ang[inl, medoid]                       # rotate member by phi -> matches medoid
    phi_bar = float(np.median(phi))              # ~ medoid's own tilt if tilts centre on 0
    upright = np.stack([rotate_canonical(X[i], float(p - phi_bar), margin=margin) for i, p in zip(inl, phi)])
    tmpl = canonicalize(upright.mean(0), X.shape[1], margin)
    return tmpl, inl, {"medoid_tilt_estimate": round(phi_bar, 1),
                       "tilt_estimate_stderr": round(float(np.std(phi) / np.sqrt(len(phi))), 2),
                       "outliers_removed": int(len(members) - len(inl))}


def montage(X, groups, templates, cell=40, per_row=20):
    rows = []
    for members, tmpl in zip(groups, templates):
        tiles = [tmpl] + [X[i] for i in members[:per_row]]
        tiles += [np.zeros_like(tmpl)] * (per_row + 1 - len(tiles))
        row = np.hstack([cv2.copyMakeBorder((255 - t * 255).astype(np.uint8), 1, 1, 1, 1,
                                            cv2.BORDER_CONSTANT, value=160) for t in tiles])
        rows.append(row)
    return np.vstack(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    t = cfg["templates"]
    margin = cfg["preprocess"]["margin"]
    rng = np.random.default_rng(t["seed"])

    idx = pd.read_csv(rpath("data/dataset/index.csv"), dtype={"captcha_id": str}, keep_default_na=False)
    train = idx[idx.split == "train"].reset_index(drop=True)
    cache = canon_for(cfg, train.crop_path)
    train = train[cache["ok"]].reset_index(drop=True)
    canon = cache["canon"][cache["ok"]].astype(np.float32)
    sizes = cache["size"][cache["ok"]]

    out = rpath(cfg["paths"]["templates_dir"])
    for p in out.glob("*"):
        if p.is_dir():
            shutil.rmtree(p)
    (out / "_review").mkdir(parents=True, exist_ok=True)

    images, labels, shape_ids, tsizes, report = [], [], [], [], {}
    for ch in cfg["captcha"]["alphabet"]:
        sel = np.nonzero(train.label.to_numpy() == ch)[0]
        if len(sel) == 0:
            report[ch] = {"status": "missing", "n": 0}
            continue
        if len(sel) > t["max_samples_per_char"]:
            sel = np.sort(rng.choice(sel, t["max_samples_per_char"], replace=False))
        X = canon[sel]
        S, ang = pairwise(X, cfg)
        S = np.maximum(S, S.T)
        D = np.clip(1 - S, 0, None)
        np.fill_diagonal(D, 0)
        rep = {"n": int(len(sel))}

        groups = [np.arange(len(sel))]
        if len(sel) < t["min_samples"]:
            rep["status"] = "insufficient_data"
        else:
            cl = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(D)
            sil2 = float(silhouette_score(D, cl, metric="precomputed"))
            minority = float(min(np.mean(cl == 0), np.mean(cl == 1)))
            rep.update(silhouette_k2=round(sil2, 3), minority_frac=round(minority, 3))
            if len(sel) >= 12:
                cl3 = AgglomerativeClustering(n_clusters=3, metric="precomputed", linkage="average").fit_predict(D)
                sil3 = float(silhouette_score(D, cl3, metric="precomputed"))
                rep["silhouette_k3"] = round(sil3, 3)
                if sil3 > sil2 + 0.1:
                    rep["warning"] = "k=3 fits better: label noise, segmentation errors or >2 shapes"
            if sil2 >= t["min_silhouette"] and minority >= t["min_minority_frac"]:
                rep["status"] = "two_shapes"
                groups = sorted([np.nonzero(cl == c)[0] for c in (0, 1)], key=len, reverse=True)
                # does one (captcha_id, pos) always render the same shape?
                key = (train.captcha_id.to_numpy()[sel] + ":" + train.pos.astype(str).to_numpy()[sel])
                same = [cl[i] == cl[j] for i in range(len(sel)) for j in range(i + 1, len(sel)) if key[i] == key[j]]
                if same:
                    rep["same_idpos_same_shape_frac"] = round(float(np.mean(same)), 3)
            else:
                rep["status"] = "one_shape_or_unseparable"

        tmpls, details = [], []
        for k, members in enumerate(groups, 1):
            tmpl, inl, info = make_template(X, members, S, ang, margin)
            tmpls.append(tmpl)
            images.append(tmpl)
            labels.append(ch)
            shape_ids.append(k)
            tsizes.append(float(np.median(sizes[sel][inl])))
            details.append({"shape": k, "members": int(len(members)), **info})
            d = out / label_dir(ch)
            d.mkdir(exist_ok=True)
            cv2.imwrite(str(d / f"shape_{k}.png"), (255 - tmpl * 255).astype(np.uint8))
            cv2.imwrite(str(d / f"shape_{k}_x4.png"), cv2.resize((255 - tmpl * 255).astype(np.uint8), None,
                                                                 fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        rep["shapes"] = details
        report[ch] = rep
        cv2.imwrite(str(out / "_review" / f"{label_dir(ch)}.png"), montage(X, groups, tmpls))
        log.info("%s: n=%d %s %s", ch, len(sel), rep["status"],
                 f"sil={rep.get('silhouette_k2')}" if "silhouette_k2" in rep else "")

    TemplateSet(np.stack(images), np.array(labels), np.array(shape_ids), np.array(tsizes, np.float32)).save(cfg)
    status = pd.Series({c: r["status"] for c, r in report.items()}).value_counts().to_dict()
    summary = {"templates": len(images), "status_counts": status,
               "expected": f"{len(cfg['captcha']['alphabet'])} chars x {t['shapes_per_char']} shapes"}
    (out / "report.json").write_text(json.dumps({"summary": summary, "per_char": report}, indent=2))
    lines = ["| char | n | status | silhouette k=2 | minority | shapes (members) | note |", "|---|---|---|---|---|---|---|"]
    for c, r in report.items():
        shapes = ", ".join(str(s["members"]) for s in r.get("shapes", []))
        lines.append(f"| `{c}` | {r['n']} | {r['status']} | {r.get('silhouette_k2', '')} | "
                     f"{r.get('minority_frac', '')} | {shapes} | {r.get('warning', '')} |")
    (out / "report.md").write_text("# Template discovery report\n\n" + json.dumps(summary) + "\n\n" + "\n".join(lines))
    log.info("%s", summary)
    unsure = [c for c, r in report.items() if r["status"] != "two_shapes"]
    if unsure:
        log.warning("two shapes NOT reliably separated for: %s (see templates/report.md)", "".join(unsure))


if __name__ == "__main__":
    main()
