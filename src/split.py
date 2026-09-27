"""Part 3 - split by captcha_id (never by image) and organise the labelled character crops.

Assignment is a stable hash of (seed, captcha_id), so ids keep their split when more
ids are labelled later - a test id can never drift into train between runs.

    data/dataset/{train,val,test}/{upper_A,lower_a,digit_7,...}/{id}_{req}_{pos}.png
    data/dataset/index.csv

    python src/split.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import label_dir, load_answers, load_config, log, rpath, setup_logging  # noqa: E402


def assign_split(captcha_id: str, cfg) -> str:
    s = cfg["split"]
    h = int(hashlib.sha256(f"{s['seed']}:{captcha_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < s["train"]:
        return "train"
    return "val" if h < s["train"] + s["val"] else "test"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--no-copy", action="store_true", help="only write index/splits, no image copies")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)

    answers = load_answers(cfg)
    if answers.empty:
        raise SystemExit("no answers yet - run src/collect.py first")
    idx = pd.read_csv(rpath(cfg["paths"]["chars_index"]), dtype={"captcha_id": str}, keep_default_na=False)
    splits = {cid: assign_split(cid, cfg) for cid in answers.captcha_id}
    rpath(cfg["paths"]["splits"]).write_text(json.dumps(
        {"method": "sha256(seed:captcha_id)", "ratios": cfg["split"], "assignment": splits}, indent=1))

    ans = answers.set_index("captcha_id")
    d = idx[idx.captcha_id.isin(splits)].copy()
    d["label"] = [ans.answer[c][p] for c, p in zip(d.captcha_id, d.pos)]
    d["split"] = d.captcha_id.map(splits)

    root = rpath("data/dataset")
    if root.exists():
        shutil.rmtree(root)
    paths = []
    for r in d.itertuples(index=False):
        dst = root / r.split / label_dir(r.label) / Path(r.crop_path).name
        if not args.no_copy:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rpath(r.crop_path), dst)
        paths.append(str(dst.relative_to(rpath("."))))
    d["dataset_path"] = paths
    root.mkdir(parents=True, exist_ok=True)
    d.to_csv(root / "index.csv", index=False)

    # sanity: no id may appear in two splits
    assert d.groupby("captcha_id").split.nunique().max() == 1
    ids = pd.Series(splits).value_counts()
    crops = d.split.value_counts()
    log.info("ids per split: %s | crops per split: %s", ids.to_dict(), crops.to_dict())
    missing = sorted(set(cfg["captcha"]["alphabet"]) - set(d[d.split == "train"].label))
    if missing:
        log.warning("characters with NO training examples yet: %s", "".join(missing))


if __name__ == "__main__":
    main()
