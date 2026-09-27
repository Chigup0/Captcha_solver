/*
 * In-browser port of the Python recognizer (segment.py "components" -> preprocess.py -> matcher.py "ncc").
 * Pure functions, no DOM: CaptchaRecognizer.create(modelJson).predict(rgba, width, height).
 * model.json (written by src/export_extension.py) holds the upright base templates; the bank of
 * rotated + blurred templates is built here at load time, exactly like matcher.NCCMatcher.
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ helpers
  function median(values) {
    const a = Float64Array.from(values).sort();
    const n = a.length;
    if (!n) return 0;
    return n % 2 ? a[(n - 1) >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2;
  }

  function fromRGBA(rgba, w, h) {
    const n = w * h, r = new Uint8Array(n), g = new Uint8Array(n), b = new Uint8Array(n);
    for (let i = 0; i < n; i++) { r[i] = rgba[4 * i]; g[i] = rgba[4 * i + 1]; b[i] = rgba[4 * i + 2]; }
    return { w, h, r, g, b };
  }

  // median of a 2 px frame (same pixel order/duplicates as numpy's concatenate in preprocess.py)
  function backgroundColor(im, border = 2) {
    const { w, h } = im, idx = [];
    for (let y = 0; y < border; y++) for (let x = 0; x < w; x++) idx.push(y * w + x);
    for (let y = h - border; y < h; y++) for (let x = 0; x < w; x++) idx.push(y * w + x);
    for (let y = 0; y < h; y++) for (let x = 0; x < border; x++) idx.push(y * w + x);
    for (let y = 0; y < h; y++) for (let x = w - border; x < w; x++) idx.push(y * w + x);
    return [median(idx.map(i => im.r[i])), median(idx.map(i => im.g[i])), median(idx.map(i => im.b[i]))];
  }

  // OpenCV's Otsu threshold (getThreshVal_Otsu_8u)
  function otsu(s8) {
    const hist = new Float64Array(256);
    for (let i = 0; i < s8.length; i++) hist[s8[i]]++;
    const scale = 1 / s8.length;
    let mu = 0;
    for (let i = 0; i < 256; i++) mu += i * hist[i];
    mu *= scale;
    let q1 = 0, mu1 = 0, maxSigma = 0, maxVal = 0;
    const EPS = 1.1920929e-7;
    for (let i = 0; i < 256; i++) {
      const p = hist[i] * scale;
      mu1 *= q1;
      q1 += p;
      const q2 = 1 - q1;
      if (Math.min(q1, q2) < EPS || Math.max(q1, q2) > 1 - EPS) continue;
      mu1 = (mu1 + i * p) / q1;
      const mu2 = (mu - q1 * mu1) / q2;
      const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
      if (sigma > maxSigma) { maxSigma = sigma; maxVal = i; }
    }
    return maxVal;
  }

  // ink = RGB distance from the background colour; binary mask via Otsu with a floor
  function inkMask(im, bg, minStrength) {
    const n = im.w * im.h, s8 = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const dr = im.r[i] - bg[0], dg = im.g[i] - bg[1], db = im.b[i] - bg[2];
      const d = Math.min(1, Math.sqrt(dr * dr + dg * dg + db * db) / 441.673);
      s8[i] = Math.floor(d * 255);
    }
    const thr = Math.max(otsu(s8), minStrength * 255);
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = s8[i] > thr ? 1 : 0;
    return mask;
  }

  // 8-connected components, labelled in raster order of their first pixel (like OpenCV)
  function components(mask, w, h) {
    const labels = new Int32Array(w * h), areas = [0], stack = [];
    let next = 1;
    for (let i = 0; i < w * h; i++) {
      if (!mask[i] || labels[i]) continue;
      labels[i] = next; stack.push(i);
      let area = 0;
      while (stack.length) {
        const p = stack.pop(); area++;
        const px = p % w, py = (p - px) / w;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const x = px + dx, y = py + dy;
          if (x < 0 || y < 0 || x >= w || y >= h) continue;
          const q = y * w + x;
          if (mask[q] && !labels[q]) { labels[q] = next; stack.push(q); }
        }
      }
      areas.push(area); next++;
    }
    return { labels, n: next, areas };
  }

  function removeSmall(mask, w, h, minArea) {
    const cc = components(mask, w, h);
    const keep = new Uint8Array(cc.n);
    let any = false;
    for (let k = 1; k < cc.n; k++) if (cc.areas[k] >= minArea) { keep[k] = 1; any = true; }
    if (cc.n > 1 && !any) {  // never delete everything: keep the largest component
      let best = 1;
      for (let k = 2; k < cc.n; k++) if (cc.areas[k] > cc.areas[best]) best = k;
      keep[best] = 1;
    }
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = keep[cc.labels[i]];
    return out;
  }

  // OpenCV 8-bit BGR2Lab (sRGB, D65), used only for colour similarity
  const LIN = new Float64Array(256).map((_, i) => {
    const v = i / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  });
  function lab(r, g, b) {
    const R = LIN[r], G = LIN[g], B = LIN[b];
    const X = (0.412453 * R + 0.35758 * G + 0.180423 * B) / 0.950456;
    const Y = 0.212671 * R + 0.71516 * G + 0.072169 * B;
    const Z = (0.019334 * R + 0.119193 * G + 0.950227 * B) / 1.088754;
    const f = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    const L = Y > 0.008856 ? 116 * Math.cbrt(Y) - 16 : 903.3 * Y;
    const c = v => Math.max(0, Math.min(255, Math.round(v)));
    return [c(L * 255 / 100), c(500 * (f(X) - f(Y)) + 128), c(200 * (f(Y) - f(Z)) + 128)];
  }

  // ------------------------------------------------------------------ segmentation ("components")
  function segment(im, P) {
    const { w, h } = im, N = P.n_chars;
    const bg = backgroundColor(im);
    const mask = removeSmall(inkMask(im, bg, P.min_strength), w, h, P.seg_min_area);
    const cc = components(mask, w, h);
    if (cc.n <= 1) return null;
    const L = new Float32Array(w * h), A = new Float32Array(w * h), B = new Float32Array(w * h);
    let groups = Array.from({ length: cc.n - 1 }, () => []);
    for (let i = 0; i < w * h; i++) {
      if (!cc.labels[i]) continue;
      groups[cc.labels[i] - 1].push(i);
      const v = lab(im.r[i], im.g[i], im.b[i]);
      L[i] = v[0]; A[i] = v[1]; B[i] = v[2];
    }
    const xOf = i => i % w;
    const color = g => [median(g.map(i => L[i])) * 0.5, median(g.map(i => A[i])), median(g.map(i => B[i]))];

    const colorDist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    const pixelGap = (a, b) => {  // distance between the closest pixels of two blobs
      let best = Infinity;
      for (const i of a) {
        const xi = xOf(i), yi = (i - xi) / w;
        for (const j of b) {
          const xj = xOf(j), yj = (j - xj) / w, d = (xi - xj) ** 2 + (yi - yj) ** 2;
          if (d < best) best = d;
        }
      }
      return Math.sqrt(best);
    };

    // too many pieces: merge until there are N
    while (groups.length > N) {
      const cols = groups.map(color);
      const sizes = groups.map(g => g.length);
      let s = 0;
      sizes.forEach((v, k) => { if (v < sizes[s]) s = k; });
      let bi, bj;
      if (sizes[s] < P.speck_frac * median(sizes)) {
        // a speck (i/j dot) is never a glyph: attach it to the similar-coloured blob with the closest pixel
        let best = Infinity, t = -1;
        groups.forEach((g, j) => {
          if (j === s) return;
          const cost = colorDist(cols[s], cols[j]) + P.w_gap * pixelGap(groups[s], g);
          if (cost < best) { best = cost; t = j; }
        });
        bi = Math.min(s, t); bj = Math.max(s, t);
      } else {  // broken strokes / leftovers: merge the most similar pair (colour + horizontal gap)
        const ext = groups.map(g => { let lo = Infinity, hi = -Infinity; for (const i of g) { const x = xOf(i); if (x < lo) lo = x; if (x > hi) hi = x; } return [lo, hi]; });
        let best = Infinity;
        for (let i = 0; i < groups.length; i++) for (let j = i + 1; j < groups.length; j++) {
          const gap = Math.max(0, Math.max(ext[i][0], ext[j][0]) - Math.min(ext[i][1], ext[j][1]));
          const cost = colorDist(cols[i], cols[j]) + P.w_gap * gap;
          if (cost < best) { best = cost; bi = i; bj = j; }
        }
      }
      groups[bi] = groups[bi].concat(groups[bj]);
      groups.splice(bj, 1);
    }

    // too few (touching glyphs): split the widest group in two by colour + x
    while (groups.length < N) {
      let wi = 0, wmax = -1;
      groups.forEach((g, k) => {
        let lo = Infinity, hi = -Infinity;
        for (const i of g) { const x = xOf(i); if (x < lo) lo = x; if (x > hi) hi = x; }
        if (hi - lo > wmax) { wmax = hi - lo; wi = k; }
      });
      const g = groups[wi];
      if (g.length < 4) break;
      const F = g.map(i => [L[i] * 0.5, A[i], B[i], xOf(i) * 1.5]);
      const mx = median(g.map(xOf));
      let a = g.map(i => (xOf(i) > mx ? 1 : 0));
      const same = arr => arr.every(v => v === arr[0]);
      for (let it = 0; it < 20; it++) {
        if (same(a)) break;
        const c = [0, 1].map(k => {
          const s = [0, 0, 0, 0]; let n = 0;
          F.forEach((f, t) => { if (a[t] === k) { for (let d = 0; d < 4; d++) s[d] += f[d]; n++; } });
          return s.map(v => v / n);
        });
        const nxt = F.map(f => {
          let d0 = 0, d1 = 0;
          for (let d = 0; d < 4; d++) { d0 += (f[d] - c[0][d]) ** 2; d1 += (f[d] - c[1][d]) ** 2; }
          return d1 < d0 ? 1 : 0;
        });
        if (nxt.every((v, t) => v === a[t])) break;
        a = nxt;
      }
      if (same(a)) break;
      groups.splice(wi, 1, g.filter((_, t) => a[t] === 0), g.filter((_, t) => a[t] === 1));
    }

    groups.sort((p, q) => median(p.map(xOf)) - median(q.map(xOf)));  // left -> right (stable)
    const crops = groups.slice(0, N).map(g => makeCrop(im, bg, g, P.crop_pad));
    while (crops.length < N) crops.push(null);
    return crops;
  }

  // glyph isolated on the background colour, padded bbox (segment.py make_crops)
  function makeCrop(im, bg, g, pad) {
    const { w, h } = im;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    const member = new Uint8Array(w * h);
    for (const i of g) {
      member[i] = 1;
      const x = i % w, y = (i - x) / w;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    x0 = Math.max(0, x0 - pad); x1 = Math.min(w, x1 + pad + 1);
    y0 = Math.max(0, y0 - pad); y1 = Math.min(h, y1 + pad + 1);
    const cw = x1 - x0, ch = y1 - y0, n = cw * ch;
    const out = { w: cw, h: ch, r: new Uint8Array(n), g: new Uint8Array(n), b: new Uint8Array(n) };
    const bgU8 = bg.map(v => Math.trunc(v));
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const s = (y + y0) * w + (x + x0), d = y * cw + x;
      if (member[s]) { out.r[d] = im.r[s]; out.g[d] = im.g[s]; out.b[d] = im.b[s]; }
      else { out.r[d] = bgU8[0]; out.g[d] = bgU8[1]; out.b[d] = bgU8[2]; }
    }
    return out;
  }

  // ------------------------------------------------------------------ canonical form (preprocess.py)
  function resizeTable(s, d, area) {
    // per output index: list of [source index, weight] (OpenCV INTER_AREA / INTER_LINEAR)
    const scale = s / d, tab = [];
    for (let dx = 0; dx < d; dx++) {
      const t = [];
      if (area) {
        const f1 = dx * scale, f2 = f1 + scale;
        const s1 = Math.ceil(f1), s2 = Math.floor(f2);
        const cell = Math.min(scale, s - f1);
        if (s1 - f1 > 1e-3) t.push([s1 - 1, (s1 - f1) / cell]);
        for (let sx = s1; sx < s2; sx++) t.push([sx, 1 / cell]);
        if (f2 - s2 > 1e-3) t.push([s2, Math.min(Math.min(f2 - s2, 1), cell) / cell]);
      } else {
        let fx = (dx + 0.5) * scale - 0.5, sx = Math.floor(fx);
        fx -= sx;
        if (sx < 0) { fx = 0; sx = 0; }
        if (sx >= s - 1) { fx = 0; sx = s - 1; }
        t.push([sx, 1 - fx]);
        if (fx) t.push([Math.min(sx + 1, s - 1), fx]);
      }
      tab.push(t);
    }
    return tab;
  }

  function resizeSquare(src, s, d) {
    const tab = resizeTable(s, d, s > d);
    const tmp = new Float32Array(s * d), out = new Float32Array(d * d);
    for (let y = 0; y < s; y++) for (let dx = 0; dx < d; dx++) {
      let v = 0;
      for (const [sx, wt] of tab[dx]) v += src[y * s + sx] * wt;
      tmp[y * d + dx] = v;
    }
    for (let dy = 0; dy < d; dy++) for (let x = 0; x < d; x++) {
      let v = 0;
      for (const [sy, wt] of tab[dy]) v += tmp[sy * d + x] * wt;
      out[dy * d + x] = v;
    }
    return out;
  }

  // tight bbox of values > thresh, pad to a centred square, resize to (S-2m)^2, place in S x S
  function canonicalize(f, w, h, S, m, thresh = 0.25) {
    const out = new Float32Array(S * S);
    let x0 = Infinity, x1 = -1, y0 = Infinity, y1 = -1;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (f[y * w + x] > thresh) {
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    if (x1 < 0) return out;
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1, side = Math.max(bw, bh);
    const sq = new Float32Array(side * side);
    const oy = (side - bh) >> 1, ox = (side - bw) >> 1;
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) sq[(y + oy) * side + x + ox] = f[(y + y0) * w + x + x0];
    const inner = S - 2 * m, r = resizeSquare(sq, side, inner);
    for (let y = 0; y < inner; y++) for (let x = 0; x < inner; x++) out[(y + m) * S + x + m] = r[y * inner + x];
    return out;
  }

  function canonical(crop, P) {
    if (!crop) return new Float32Array(P.canonical_size ** 2);
    const bg = backgroundColor(crop);
    const clean = removeSmall(inkMask(crop, bg, P.min_strength), crop.w, crop.h, P.pre_min_area);
    return canonicalize(clean, crop.w, crop.h, P.canonical_size, P.margin);
  }

  // preprocess.rotate: counter-clockwise on an enlarged canvas, cv2.warpAffine INTER_LINEAR
  // (OpenCV's 1/32-pixel fixed-point sampling), zero border; then re-canonicalize
  function rotateCanonical(img, S, angle, margin) {
    if (angle === 0) return Float32Array.from(img);
    const side = Math.ceil(Math.hypot(S, S)) + 2, off = (side - S) >> 1;
    const canvas = new Float32Array(side * side);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) canvas[(y + off) * side + x + off] = img[y * S + x];
    const rad = angle * Math.PI / 180, a = Math.cos(rad), b = Math.sin(rad), c = side / 2;
    // forward M = [[a, b, (1-a)c - bc], [-b, a, bc + (1-a)c]]; warpAffine samples src = M^-1 * dst
    const m02 = (1 - a) * c - b * c, m12 = b * c + (1 - a) * c;
    const det = a * a + b * b;
    const i00 = a / det, i01 = -b / det, i10 = b / det, i11 = a / det;
    const i02 = -(i00 * m02 + i01 * m12), i12 = -(i10 * m02 + i11 * m12);
    const out = new Float32Array(side * side);
    const px = (x, y) => (x < 0 || y < 0 || x >= side || y >= side ? 0 : canvas[y * side + x]);
    for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
      const X = Math.round((i00 * x + i01 * y + i02) * 32), Y = Math.round((i10 * x + i11 * y + i12) * 32);
      const sx = X >> 5, sy = Y >> 5, fx = (X & 31) / 32, fy = (Y & 31) / 32;
      out[y * side + x] = (1 - fy) * ((1 - fx) * px(sx, sy) + fx * px(sx + 1, sy))
                        + fy * ((1 - fx) * px(sx, sy + 1) + fx * px(sx + 1, sy + 1));
    }
    return canonicalize(out, side, side, S, margin);
  }

  function decodeBase64(s) {
    const bin = atob(s), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // GaussianBlur(sigma) on a float image: ksize = round(8*sigma+1)|1, BORDER_REFLECT_101
  function blur(img, S, sigma) {
    if (!(sigma > 0)) return img;
    const ks = Math.round(sigma * 8 + 1) | 1, r = ks >> 1, k = [];
    let sum = 0;
    for (let i = 0; i < ks; i++) { k.push(Math.exp(-((i - r) ** 2) / (2 * sigma * sigma))); sum += k[i]; }
    for (let i = 0; i < ks; i++) k[i] /= sum;
    const refl = i => (i < 0 ? -i : i >= S ? 2 * S - 2 - i : i);
    const tmp = new Float32Array(S * S), out = new Float32Array(S * S);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      let v = 0;
      for (let i = 0; i < ks; i++) v += img[y * S + refl(x + i - r)] * k[i];
      tmp[y * S + x] = v;
    }
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      let v = 0;
      for (let i = 0; i < ks; i++) v += tmp[refl(y + i - r) * S + x] * k[i];
      out[y * S + x] = v;
    }
    return out;
  }

  // ------------------------------------------------------------------ matcher (ncc)
  // every template rotated over the angle range and blurred -> uint8 rows (matcher.NCCMatcher's bank)
  function buildBank(model) {
    const S = model.canonical_size, F = S * S, T = model.template_labels.length;
    const tmpl = decodeBase64(model.templates);
    const angles = [];
    for (let a = -model.max_angle; a <= model.max_angle + 1e-6; a += model.angle_step) angles.push(a);
    const bank = new Uint8Array(T * angles.length * F), labels = [];
    for (let t = 0; t < T; t++) {
      const img = new Float32Array(F);
      for (let f = 0; f < F; f++) img[f] = tmpl[t * F + f] / 255;
      angles.forEach((ang, k) => {
        const row = blur(rotateCanonical(img, S, ang, model.margin), S, model.sigma);
        const off = (t * angles.length + k) * F;
        for (let f = 0; f < F; f++) bank[off + f] = Math.max(0, Math.min(255, Math.round(row[f] * 255)));
        labels.push(model.template_labels[t]);
      });
    }
    return { bank, labels, rows: labels.length, features: F };
  }

  function create(model) {
    const built = buildBank(model);
    const bank = built.bank, F = built.features, R = built.rows;
    model = Object.assign({}, model, { labels: built.labels });
    // NCC(q, row) = (q_unit . row) / ||row - mean(row)||   (q_unit is zero-mean, so row's mean drops out)
    const norms = new Float32Array(R);
    for (let r = 0; r < R; r++) {
      let s = 0, s2 = 0;
      for (let f = 0; f < F; f++) { const v = bank[r * F + f]; s += v; s2 += v * v; }
      norms[r] = Math.sqrt(Math.max(0, s2 - (s * s) / F)) + 255e-6;
    }

    function matchOne(canon) {
      const q = blur(canon, model.canonical_size, model.sigma);
      let mean = 0;
      for (let f = 0; f < F; f++) mean += q[f];
      mean /= F;
      let nrm = 0;
      const u = new Float32Array(F);
      for (let f = 0; f < F; f++) { u[f] = q[f] - mean; nrm += u[f] * u[f]; }
      nrm = Math.sqrt(nrm) + 1e-6;
      for (let f = 0; f < F; f++) u[f] /= nrm;
      const best = {};  // label -> best score
      for (let r = 0; r < R; r++) {
        let d = 0;
        const off = r * F;
        for (let f = 0; f < F; f++) d += u[f] * bank[off + f];
        const s = d / norms[r], lab = model.labels[r];
        if (!(lab in best) || s > best[lab]) best[lab] = s;
      }
      const order = Object.entries(best).sort((a, b) => b[1] - a[1]);
      return { label: order[0][0], score: order[0][1], margin: order[0][1] - (order[1] ? order[1][1] : -1) };
    }

    function predict(rgba, width, height) {
      const crops = segment(fromRGBA(rgba, width, height), model);
      if (!crops) return null;
      const m = crops.map(c => matchOne(canonical(c, model)));
      return {
        text: m.map(x => x.label).join(""),
        scores: m.map(x => x.score),
        margins: m.map(x => x.margin),
      };
    }

    return { predict };
  }

  root.CaptchaRecognizer = { create };
})(typeof globalThis !== "undefined" ? globalThis : this);
