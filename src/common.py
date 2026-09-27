"""Shared helpers: config, paths, label folder names, answers I/O."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"

log = logging.getLogger("captcha")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rpath(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


# Windows/macOS filesystems are case-insensitive, so "A/" and "a/" would collide.
def label_dir(ch: str) -> str:
    if ch.isdigit():
        return f"digit_{ch}"
    return f"upper_{ch}" if ch.isupper() else f"lower_{ch}"


# --------------------------------------------------------------------------- answers
def valid_answer(ans, cfg) -> bool:
    return (
        isinstance(ans, str)
        and len(ans) == cfg["captcha"]["n_chars"]
        and all(c in cfg["captcha"]["alphabet"] for c in ans)
    )


def load_answers(cfg) -> pd.DataFrame:
    path = rpath(cfg["paths"]["answers"])
    if not path.exists():
        return pd.DataFrame(columns=["captcha_id", "answer"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def save_answer(captcha_id: str, answer: str, cfg) -> None:
    """Add or replace one id's answer in answers.csv."""
    df = load_answers(cfg)
    df = pd.concat([df[df.captcha_id != captcha_id], pd.DataFrame([{"captcha_id": captcha_id, "answer": answer}])])
    path = rpath(cfg["paths"]["answers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("captcha_id").to_csv(path, index=False)


# --------------------------------------------------------------------------- metadata
def load_metadata(cfg) -> pd.DataFrame:
    path = rpath(cfg["paths"]["metadata"])
    if not path.exists():
        raise SystemExit(f"No metadata at {path}; run src/collect.py first.")
    return pd.read_csv(path, dtype={"captcha_id": str, "answer": str}, keep_default_na=False)


def image_path(cfg, filename: str) -> Path:
    return rpath(cfg["paths"]["raw_dir"]) / "images" / filename
