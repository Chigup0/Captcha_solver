/*
 * Finds the captcha image on the page, recognises it, and types the answer into the captcha
 * box that follows it. Runs ONLY on CAPTCHA_HOST (config.js): the manifest limits injection to
 * that host, and this script checks again before doing anything (the bookmarklet relies on
 * this check alone).
 */
(() => {
  "use strict";
  if (typeof CAPTCHA_HOST !== "string" || location.hostname !== CAPTCHA_HOST) {
    if (typeof CAPTCHA_MODEL === "object") alert(`This bookmarklet only works on ${CAPTCHA_HOST}.`);
    return;
  }
  if (window.__captchaAutofill) {  // bookmarklet clicked again: just re-solve the current image
    document.querySelectorAll("img").forEach(img => window.__captchaAutofill(img, true));
    return;
  }

  const CAPTCHA_PATH = /\/image-captcha-generate\//;
  let recognizer = null;

  // the bookmarklet carries the model inline (CAPTCHA_MODEL); the extension ships model.json
  function loadRecognizer() {
    recognizer = recognizer || (typeof CAPTCHA_MODEL === "object"
      ? Promise.resolve(CAPTCHA_MODEL)
      : fetch(chrome.runtime.getURL("model.json")).then(r => r.json())
    ).then(model => CaptchaRecognizer.create(model));
    return recognizer;
  }

  function isCaptcha(img) {
    try {
      const u = new URL(img.currentSrc || img.src, location.href);
      return u.hostname === CAPTCHA_HOST && CAPTCHA_PATH.test(u.pathname);
    } catch {
      return false;
    }
  }

  // the captcha answer box: Drupal's captcha_response field, else the first visible text box after the image
  function findInput(img) {
    const scope = img.closest("form") || document;
    const named = scope.querySelector('input[name="captcha_response"]');
    if (named) return named;
    const boxes = [...scope.querySelectorAll('input[type="text"], input:not([type])')]
      .filter(i => !i.disabled && !i.readOnly && i.offsetParent !== null);
    return boxes.find(i => img.compareDocumentPosition(i) & Node.DOCUMENT_POSITION_FOLLOWING) || null;
  }

  // set the value the way typing would, so the page's own listeners see it
  function typeInto(input, text) {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    input.focus();
    setter.call(input, text);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  const solved = new WeakMap();  // img -> src already solved (re-solve when the captcha refreshes)

  async function solve(img, force = false) {
    if (!isCaptcha(img)) return;
    if (!img.complete || !img.naturalWidth) {
      img.addEventListener("load", () => solve(img), { once: true });
      return;
    }
    const src = img.currentSrc || img.src;
    if (!force && solved.get(img) === src) return;
    solved.set(img, src);

    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    let pixels;
    try {
      pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
    } catch (e) {
      console.warn("[captcha] cannot read the image pixels:", e);
      return;
    }
    const result = (await loadRecognizer()).predict(pixels.data, canvas.width, canvas.height);
    if (!result) return;
    const input = findInput(img);
    if (!input) {
      console.warn("[captcha] predicted", result.text, "but found no input box after the image");
      return;
    }
    typeInto(input, result.text);
    console.info("[captcha]", result.text, "min score", Math.min(...result.scores).toFixed(3));
  }

  window.__captchaAutofill = solve;
  document.querySelectorAll("img").forEach(img => solve(img));
  new MutationObserver(mutations => {
    for (const m of mutations) {
      if (m.type === "attributes" && m.target.tagName === "IMG") solve(m.target);
      for (const node of m.addedNodes) {
        if (node.nodeType !== 1) continue;
        if (node.tagName === "IMG") solve(node);
        else node.querySelectorAll?.("img").forEach(img => solve(img));
      }
    }
  }).observe(document.documentElement, { subtree: true, childList: true, attributes: true, attributeFilter: ["src"] });
})();
