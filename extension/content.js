const MAX_TEXT_LENGTH = 600;
const MIN_TEXT_LENGTH = 2;
const BATCH_SIZE = 25;
const SVGNS = "http://www.w3.org/2000/svg";

const TIER_LABEL = { safe: "Safe", borderline: "Borderline", toxic: "Toxic" };

const fallbackTerms = [
  "aggressive", "attack", "bully", "hate", "hurt", "insult", "kill",
  "moderator", "not cool", "stupid", "threat", "harass", "idiot",
  "loser", "trash", "garbage", "disgusting", "worthless"
];

// Cache predictions per node so React re-renders can re-attach without re-fetching.
const predCache = new WeakMap();

let scanning = false;
let rescanQueued = false;
let totalScanned = 0;
let totalFlagged = 0;
let lastSource = "raven-api";

console.info("%c[Raven]%c content script loaded on " + location.host, "color:#258dff;font-weight:bold", "");

function tierFor(prediction) {
  const score = prediction.score || 0;
  if (prediction.needs_review || score >= 0.5) return "toxic";
  if (score >= 0.35) return "borderline";
  return "safe";
}

function fallbackPredict(text) {
  const normalized = text.toLowerCase();
  const hits = fallbackTerms.filter((term) => normalized.includes(term)).length;
  const score = Math.min(0.96, 0.08 + hits * 0.3 + Math.min(text.length / 600, 0.12));
  return {
    needs_review: hits > 0 || score >= 0.5,
    score,
    label: hits > 0 || score >= 0.5 ? "review" : "safe",
    source: "extension-demo-fallback"
  };
}

async function predictBatch(texts) {
  try {
    const response = await chrome.runtime.sendMessage({ type: "RAVEN_PREDICT", texts });
    if (response?.ok && Array.isArray(response.predictions)) return response.predictions;
    throw new Error(response?.error || "no predictions");
  } catch {
    return texts.map(fallbackPredict);
  }
}

// Built with DOM APIs (no innerHTML) so it survives Trusted Types CSP on x.com.
function svgPath(d, fill) {
  const path = document.createElementNS(SVGNS, "path");
  path.setAttribute("d", d);
  path.setAttribute("fill", fill);
  return path;
}

function ravenMark() {
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("class", "raven-pill-mark");
  svg.setAttribute("viewBox", "0 0 96 96");
  svg.setAttribute("aria-hidden", "true");
  svg.appendChild(svgPath("M48 18L70 27V45.5C70 59.3 61.8 71.9 48 78C34.2 71.9 26 59.3 26 45.5V27L48 18Z", "#ffffff"));
  svg.appendChild(svgPath("M55 32 L38 53 H47 L43 65 L60 43 H51 L55 32 Z", "rgba(8,30,60,0.42)"));
  return svg;
}

function buildPill(prediction) {
  const tier = tierFor(prediction);
  const percent = Math.round((prediction.score || 0) * 100);

  const pill = document.createElement("span");
  pill.className = `raven-pill raven-${tier}`;
  pill.title = `Raven · ${TIER_LABEL[tier]} · ${percent}% toxic`;

  const label = document.createElement("span");
  label.className = "raven-pill-text";
  label.textContent = TIER_LABEL[tier];

  const pct = document.createElement("span");
  pct.className = "raven-pill-pct";
  pct.textContent = `${percent}%`;

  pill.append(ravenMark(), label, pct);
  pill.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  return { pill, tier };
}

function attachPill(node, header, prediction) {
  if (!header || header.querySelector(":scope > .raven-pill")) return false;
  const { pill, tier } = buildPill(prediction);
  header.appendChild(pill);
  node.classList.add("raven-scanned");
  node.classList.toggle("raven-flagged", tier === "toxic");
  return tier === "toxic";
}

function findUnits() {
  const host = location.hostname;
  if (host.includes("youtube")) {
    return Array.from(document.querySelectorAll("ytd-comment-view-model, ytd-comment-renderer")).map((node) => {
      const author = node.querySelector("#header-author") || node.querySelector("#author-text");
      const textNode = node.querySelector("#content-text");
      return {
        node,
        header: author ? author.parentElement || author : null,
        text: textNode ? textNode.textContent.trim() : ""
      };
    });
  }
  return Array.from(document.querySelectorAll('article[data-testid="tweet"]')).map((node) => {
    const textNode = node.querySelector('[data-testid="tweetText"]');
    return {
      node,
      header: node.querySelector('[data-testid="User-Name"]'),
      text: textNode ? textNode.textContent.trim() : ""
    };
  });
}

// One pass: re-attach cached pills + score every currently-unscanned post.
async function runScanOnce() {
  const units = findUnits();
  const toFetch = [];

  for (const unit of units) {
    if (!unit.header) continue;
    if (predCache.has(unit.node)) {
      // Re-attach if a React re-render wiped our pill.
      attachPill(unit.node, unit.header, predCache.get(unit.node));
      continue;
    }
    if (unit.text.length >= MIN_TEXT_LENGTH && unit.text.length <= MAX_TEXT_LENGTH) {
      toFetch.push(unit);
    }
  }

  let flagged = 0;
  for (let i = 0; i < toFetch.length; i += BATCH_SIZE) {
    const chunk = toFetch.slice(i, i + BATCH_SIZE);
    const predictions = await predictBatch(chunk.map((u) => u.text));
    chunk.forEach((unit, index) => {
      const prediction = predictions[index] || fallbackPredict(unit.text);
      lastSource = prediction.source || lastSource;
      predCache.set(unit.node, prediction);
      if (attachPill(unit.node, unit.header, prediction)) flagged += 1;
    });
  }

  if (toFetch.length) {
    totalScanned += toFetch.length;
    totalFlagged += flagged;
    console.info(`[Raven] scored ${toFetch.length} new post(s) · ${flagged} toxic · ${units.length} on page`);
    try {
      chrome.storage?.local?.set({ ravenLastScan: summary(toFetch.length, flagged) });
    } catch {
      /* ignore */
    }
  }
  return toFetch.length;
}

// Drain loop: keeps scanning while more scans are requested (e.g. mid-scroll),
// so a scan triggered during an in-flight scan is never dropped.
async function scanPage() {
  if (scanning) {
    rescanQueued = true;
    return summary();
  }
  scanning = true;
  try {
    do {
      rescanQueued = false;
      await runScanOnce();
    } while (rescanQueued);
  } finally {
    scanning = false;
  }
  return summary();
}

function summary(lastScanned = 0, lastFlagged = 0) {
  return {
    scanned: totalScanned,
    flagged: totalFlagged,
    lastScanned,
    lastFlagged,
    source: lastSource,
    updatedAt: new Date().toISOString()
  };
}

let scanTimer = 0;
function scheduleScan() {
  window.clearTimeout(scanTimer);
  scanTimer = window.setTimeout(scanPage, 450);
}

const observer = new MutationObserver(scheduleScan);
observer.observe(document.documentElement, { childList: true, subtree: true });
window.addEventListener("scroll", scheduleScan, { passive: true });

// Initial scans (the timeline loads after document_idle).
scanPage();
[800, 2000, 4000].forEach((delay) => window.setTimeout(scanPage, delay));

// Safety net: catch any post the observer/scroll missed (cheap — does nothing
// when there's nothing new to score).
window.setInterval(scanPage, 2000);

// Re-scan on X's client-side route changes (SPA navigation, no reload).
let lastUrl = location.href;
window.setInterval(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    scheduleScan();
  }
}, 1500);

chrome.runtime?.onMessage?.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "RAVEN_SCAN_NOW") return false;
  scanPage()
    .then((result) => sendResponse({ ok: true, summary: result }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));
  return true;
});
