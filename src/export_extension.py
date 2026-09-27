"""Step 5 - build the browser extension and the bookmarklet from the trained templates.

Writes into extension/ (next to the hand-written recognizer.js and content.js):
    manifest.json   runs only on api.host from config.yaml (http + https, no subdomains)
    config.js       the same host, checked again at runtime
    model.json      recogniser settings + the upright base templates (uint8, base64)
    icons/          plain 16/48/128 px icon
and into dist/:
    captcha-extension.zip   the extension, manifest at the zip root
    bookmarklet.txt         javascript: URL (model inlined, same host lock)
    bookmarklet.html        open it and drag the button to the bookmarks bar

    python src/export_extension.py

Re-run after every retrain (then reload the extension / re-add the bookmarklet).
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import load_config, log, rpath, setup_logging  # noqa: E402
from matcher import TemplateSet  # noqa: E402

ICON_SIZES = (16, 48, 128)
EXT_FILES = ["manifest.json", "config.js", "recognizer.js", "content.js", "model.json"] + \
            [f"icons/icon{s}.png" for s in ICON_SIZES]


def write_icons(out: Path) -> dict:
    """Plain icon: blue square, white "C", drawn at 128 px and downscaled."""
    img = np.zeros((128, 128, 4), np.uint8)
    cv2.rectangle(img, (8, 8), (119, 119), (200, 110, 40, 255), -1)
    (tw, th), _ = cv2.getTextSize("C", cv2.FONT_HERSHEY_DUPLEX, 3.2, 8)
    cv2.putText(img, "C", ((128 - tw) // 2, (128 + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 3.2,
                (255, 255, 255, 255), 8, cv2.LINE_AA)
    (out / "icons").mkdir(exist_ok=True)
    icons = {}
    for s in ICON_SIZES:
        cv2.imwrite(str(out / "icons" / f"icon{s}.png"), cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA))
        icons[str(s)] = f"icons/icon{s}.png"
    return icons


def build_model(cfg) -> dict:
    t = TemplateSet.load(cfg)
    m, p, seg = cfg["matching"], cfg["preprocess"], cfg["segmentation"]
    tmpl = np.clip(np.round(t.images * 255), 0, 255).astype(np.uint8)
    return {
        "template_labels": "".join(t.labels),
        "templates": base64.b64encode(tmpl.tobytes()).decode(),
        "max_angle": m["max_rotation"] + m["rotation_margin"], "angle_step": m["rotation_step"],
        "canonical_size": p["canonical_size"], "margin": p["margin"], "sigma": p["blur_sigma"],
        "min_strength": seg["min_strength"], "seg_min_area": seg["min_component_area"],
        "pre_min_area": p["min_component_area"], "crop_pad": seg["crop_pad"],
        "n_chars": cfg["captcha"]["n_chars"], "w_gap": seg.get("components", {}).get("w_gap", 2.0),
        "speck_frac": seg.get("components", {}).get("speck_frac", 0.15),
    }


def write_bookmarklet(ext: Path, dist: Path, host: str, model: dict) -> int:
    code = ("(function(){"
            f"var CAPTCHA_HOST={json.dumps(host)};"
            f"if(location.hostname!==CAPTCHA_HOST){{alert('This bookmarklet only works on '+CAPTCHA_HOST+'.');return;}}"
            f"var CAPTCHA_MODEL={json.dumps(model, separators=(',', ':'))};\n"
            + (ext / "recognizer.js").read_text(encoding="utf-8") + "\n"
            + (ext / "content.js").read_text(encoding="utf-8") + "\n})();")
    url = "javascript:" + quote(code, safe="")
    (dist / "bookmarklet.txt").write_text(url, encoding="utf-8")
    (dist / "bookmarklet.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Captcha autofill bookmarklet</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}}
a.bm{{display:inline-block;padding:10px 18px;background:#2866c8;color:#fff;border-radius:8px;
text-decoration:none;font-weight:600}}code{{background:#eee;padding:1px 4px;border-radius:4px}}</style></head>
<body><h1>Captcha autofill bookmarklet</h1>
<p>Drag this button to your bookmarks bar:</p>
<p><a class="bm" href="{html.escape(url)}">Captcha autofill</a></p>
<p>Then open the login page on <code>{html.escape(host)}</code> and click the bookmark. It reads the captcha and
types the answer into the captcha box. On any other site it only shows a message and does nothing.
Click it again after the captcha refreshes (it also follows refreshes by itself while the page stays open).</p>
</body></html>""", encoding="utf-8")
    return len(url)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    setup_logging()
    cfg = load_config(args.config)

    host = str(cfg["api"]["host"]).strip().removeprefix("https://").removeprefix("http://").strip("/").lower()
    if not re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", host):
        raise SystemExit(f"api.host must be a plain domain like abc.in, got {host!r}")
    if cfg["segmentation"]["method"] != "components":
        raise SystemExit("the extension implements the 'components' segmenter only; set segmentation.method")

    ext, dist = rpath("extension"), rpath("dist")
    ext.mkdir(exist_ok=True)
    dist.mkdir(exist_ok=True)
    (ext / "model.bin").unlink(missing_ok=True)  # old format (pre-rotated bank)
    model = build_model(cfg)
    (ext / "model.json").write_text(json.dumps(model))
    (ext / "config.js").write_text(f'// generated by src/export_extension.py - do not edit\n'
                                   f'const CAPTCHA_HOST = {json.dumps(host)};\n')
    matches = [f"https://{host}/*", f"http://{host}/*"]
    manifest = {
        "manifest_version": 3,
        "name": f"Captcha autofill ({host})",
        "version": "1.1",
        "description": f"Reads the captcha on {host} and types it into the captcha box. Runs on no other site.",
        "icons": write_icons(ext),
        "content_scripts": [{"matches": matches, "js": ["config.js", "recognizer.js", "content.js"],
                             "run_at": "document_idle", "all_frames": False}],
        "web_accessible_resources": [{"resources": ["model.json"], "matches": matches}],
    }
    (ext / "manifest.json").write_text(json.dumps(manifest, indent=2))

    with zipfile.ZipFile(dist / "captcha-extension.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in EXT_FILES:
            z.write(ext / f, f)
    n = write_bookmarklet(ext, dist, host, model)
    log.info("host %s: %d templates | extension zip %.0f KB | bookmarklet %.0f KB -> %s",
             host, len(model["template_labels"]), (dist / "captcha-extension.zip").stat().st_size / 1e3, n / 1e3, dist)


if __name__ == "__main__":
    main()
