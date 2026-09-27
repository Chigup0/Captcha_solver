"""Step 1 - collect labelled captchas (and check the model while you do it).

For each captcha_id (start, start+1, start+2, ...):
  1. fetch one rendering, show it, and - once templates exist - show the model's prediction
  2. you confirm it or type the correct 5-character answer
  3. if the prediction was wrong (or there was none), `--per-id` renderings of that id are
     saved, all labelled with the correct answer; correct predictions are only logged

At the prompt:
    Enter   the prediction is correct
    A7kP2   the correct answer (when the prediction is wrong, or there is none yet)
    s       skip this id
    r       show another rendering (if this one is hard to read)
    q       quit

Saved after every id; the next run continues after the last id handled.

    python src/collect.py --start 1559187 --per-id 20

Output:
    data/raw/images/<id>_<req>.jpg
    data/raw/metadata.csv       captcha_id, request_no, filename, answer, ...
    data/labels/answers.csv     captcha_id, answer
    data/raw/predictions.csv    captcha_id, predicted, answer, correct (model check log)
    data/raw/skipped.txt        ids you skipped
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, log, rpath, save_answer, setup_logging, valid_answer  # noqa: E402

META_COLUMNS = ["captcha_id", "request_no", "filename", "answer", "sha1", "http_status", "url_nonce", "fetched_at"]


class Client:
    def __init__(self, cfg: dict):
        self.api = api = cfg["api"]
        host = str(api.get("host") or "").strip().removeprefix("https://").removeprefix("http://").strip("/")
        if not host:
            raise SystemExit("Set api.host in config/config.yaml (e.g. host: abc.in).")
        self.fields = {"host": host, "base_url": f"{api.get('scheme', 'https')}://{host}"}
        for field, env in (api.get("template_fields_env") or {}).items():
            if env not in os.environ:
                raise SystemExit(f"url_template field {{{field}}} needs env var {env}")
            self.fields[field] = os.environ[env]
        self.session = requests.Session()
        self.session.headers.update({k: str(v).format(**self.fields) for k, v in (api.get("headers") or {}).items()})
        for header, env in (api.get("header_env") or {}).items():
            if os.environ.get(env):
                self.session.headers[header] = os.environ[env]
        self._last = 0.0
        self._last_nonce = 0

    def _nonce(self) -> str:
        if self.api.get("nonce") == "fixed":
            return str(self.api["nonce_fixed"])
        # unix time, strictly increasing so two requests in one second never share it
        self._last_nonce = max(int(time.time()), self._last_nonce + 1)
        return str(self._last_nonce)

    def fetch(self, captcha_id: str) -> tuple[bytes, str, int]:
        nonce = self._nonce()
        url = self.api["url_template"].format(captcha_id=captcha_id, nonce=nonce, **self.fields)
        for attempt in range(self.api.get("max_retries", 3) + 1):
            wait = self.api.get("min_interval_s", 0.5) - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, timeout=self.api.get("timeout_s", 10))
            except requests.RequestException as e:
                log.warning("%s (attempt %d)", e, attempt + 1)
            else:
                if r.status_code == 200:
                    return r.content, nonce, r.status_code
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()  # 4xx: retrying will not help
                log.warning("HTTP %s for %s (attempt %d)", r.status_code, url, attempt + 1)
            time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"giving up on {url}")


def image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "bin"


def decode(data: bytes):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


_window = None  # (tk root, label), created on first show()


def show(img: np.ndarray) -> None:
    """Show the captcha 4x enlarged in a small always-on-top window (tkinter, so it also
    works with opencv-python-headless)."""
    global _window
    big = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    try:
        import tkinter as tk
        if _window is None:
            root = tk.Tk()
            root.title("captcha")
            root.attributes("-topmost", True)
            label = tk.Label(root)
            label.pack()
            _window = (root, label)
        root, label = _window
        photo = tk.PhotoImage(data=base64.b64encode(cv2.imencode(".png", big)[1].tobytes()))
        label.configure(image=photo)
        label.image = photo  # keep a reference or Tk drops the image
        root.update()
    except Exception:  # no display available: fall back to a file
        path = rpath("data/raw/_current.png")
        cv2.imwrite(str(path), big)
        print(f"  (open {path} to see it)")


def close_window() -> None:
    global _window
    if _window is not None:
        try:
            _window[0].destroy()
        except Exception:
            pass
        _window = None


def save_id(client: Client, cfg, sid: str, answer: str, first: tuple, per_id: int) -> int:
    img_dir = rpath(cfg["paths"]["raw_dir"]) / "images"
    meta_path = rpath(cfg["paths"]["metadata"])
    rows, hashes = [], set()
    for req in range(per_id):
        data, nonce, status = first if req == 0 else client.fetch(sid)
        if decode(data) is None:
            log.warning("id %s req %d: not an image, skipped", sid, req)
            continue
        fname = f"{sid}_{req:03d}.{image_ext(data)}"
        (img_dir / fname).write_bytes(data)
        sha1 = hashlib.sha1(data).hexdigest()
        hashes.add(sha1)
        rows.append(dict(captcha_id=sid, request_no=req, filename=fname, answer=answer, sha1=sha1,
                         http_status=status, url_nonce=nonce,
                         fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    pd.DataFrame(rows, columns=META_COLUMNS).to_csv(meta_path, mode="a", header=not meta_path.exists(),
                                                    index=False)
    save_answer(sid, answer, cfg)
    print(f"  saved {len(rows)} images ({len(hashes)} distinct)")
    return len(rows)


def load_model(cfg):
    """The trained recognizer, or None until build_templates.py has been run."""
    if not (rpath(cfg["paths"]["templates_dir"]) / "templates.npz").exists():
        print("no templates yet - type answers by hand (run segment/split/build_templates to get predictions)")
        return None
    from predict import Pipeline
    return Pipeline(cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--start", type=int, help="first captcha_id (default: continue after the last one handled)")
    ap.add_argument("--per-id", type=int, help="renderings saved per id (default: collection.per_id)")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)
    per_id = args.per_id or cfg["collection"]["per_id"]

    raw = rpath(cfg["paths"]["raw_dir"])
    (raw / "images").mkdir(parents=True, exist_ok=True)
    meta_path = rpath(cfg["paths"]["metadata"])
    skipped_path = raw / "skipped.txt"
    handled = set()
    if meta_path.exists():
        handled |= set(pd.read_csv(meta_path, dtype={"captcha_id": str}).captcha_id)
    if skipped_path.exists():
        handled |= set(skipped_path.read_text().split())
    if (raw / "predictions.csv").exists():  # correctly predicted ids have no images, only this log
        handled |= set(pd.read_csv(raw / "predictions.csv", dtype={"captcha_id": str}).captcha_id)

    if args.start is not None:
        cid = args.start
    else:
        done = [int(x) for x in handled if x.isdigit()]
        cid = max(done) + 1 if done else cfg["collection"]["start"]

    client = Client(cfg)
    model = load_model(cfg)
    pred_path = raw / "predictions.csv"
    n_ids = n_imgs = n_checked = n_correct = 0
    try:
        while True:
            sid = str(cid)
            cid += 1
            if sid in handled:
                continue
            first = client.fetch(sid)
            img = decode(first[0])
            if img is None:
                print(f"[{sid}] response is not an image - skipping")
                continue
            while True:
                show(img)
                pred = model.predict(img)["text"] if model else None
                if pred:
                    ans = input(f"[{sid}] predicted: {pred}   Enter=correct / type answer / s=skip / r=another"
                                f" / q=quit: ").strip()
                else:
                    ans = input(f"[{sid}] answer / s=skip / r=another / q=quit: ").strip()
                if ans == "q":
                    return
                if ans == "r":
                    new = client.fetch(sid)
                    if decode(new[0]) is not None:
                        first, img = new, decode(new[0])
                    continue
                if ans == "s":
                    with open(skipped_path, "a") as f:
                        f.write(sid + "\n")
                    break
                if ans == "" and pred:
                    ans = pred
                if not valid_answer(ans, cfg):
                    print(f"  need exactly {cfg['captcha']['n_chars']} characters from A-Z a-z 0-9")
                    continue
                if pred:
                    n_checked += 1
                    n_correct += pred == ans
                    if pred != ans:
                        print("  wrong at:  " + " ".join(f"{p}->{a}" for p, a in zip(pred, ans) if p != a))
                    print(f"  model this session: {n_correct}/{n_checked} correct ({n_correct / n_checked:.0%})")
                    pd.DataFrame([dict(captcha_id=sid, predicted=pred, answer=ans, correct=pred == ans)]).to_csv(
                        pred_path, mode="a", header=not pred_path.exists(), index=False)
                    if pred == ans:  # model already knows this one: nothing new to learn, don't save
                        break
                n_imgs += save_id(client, cfg, sid, ans, first, per_id)
                n_ids += 1
                break
    except KeyboardInterrupt:
        print()
    finally:
        close_window()
        print(f"this session: {n_ids} ids labelled, {n_imgs} images saved -> {meta_path}")
        if n_checked:
            print(f"model: {n_correct}/{n_checked} full answers correct ({n_correct / n_checked:.0%})")


if __name__ == "__main__":
    main()
