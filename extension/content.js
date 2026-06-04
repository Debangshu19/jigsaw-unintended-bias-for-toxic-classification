const RAVEN_API = "http://localhost:8000/predict-batch";
const MAX_TEXT_LENGTH = 600;
const SCAN_LIMIT = 60;

const selectors = [
  "ytd-comment-renderer #content-text",
  "[data-testid='tweetText']",
  "article div[lang]",
  "ul li span",
  "[role='article'] span"
];

const fallbackTerms = [
  "aggressive",
  "attack",
  "bully",
  "hate",
  "hurt",
  "insult",
  "moderator",
  "not cool",
  "review",
  "reviewed",
  "stupid",
  "threat",
  "harass"
];

function getCandidates() {
  const nodes = selectors.flatMap((selector) => Array.from(document.querySelectorAll(selector)));
  const unique = [...new Set(nodes)];

  return unique
    .map((node) => ({ node, text: node.textContent.trim() }))
    .filter(({ node, text }) => {
      if (!text || text.length < 16 || text.length > MAX_TEXT_LENGTH) return false;
      if (node.dataset.ravenScanned === "true") return false;
      return node.offsetParent !== null;
    })
    .slice(0, SCAN_LIMIT);
}

function fallbackPredict(text) {
  const normalized = text.toLowerCase();
  const hits = fallbackTerms.filter((term) => normalized.includes(term)).length;
  const score = Math.min(0.95, 0.12 + hits * 0.3 + Math.min(text.length / 500, 0.16));
  return {
    needs_review: hits > 0 || score >= 0.58,
    score,
    label: hits > 0 || score >= 0.58 ? "review" : "safe",
    source: "extension-demo-fallback"
  };
}

async function predictBatch(texts) {
  try {
    const response = await fetch(RAVEN_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts })
    });

    if (!response.ok) throw new Error("Raven API unavailable");
    const data = await response.json();
    return data.predictions;
  } catch {
    return texts.map(fallbackPredict);
  }
}

function markNode(node, prediction) {
  node.dataset.ravenScanned = "true";
  if (!prediction.needs_review) return false;

  node.classList.add("raven-needs-review");
  node.title = `Raven review: ${Math.round(prediction.score * 100)}%`;

  const badge = document.createElement("span");
  badge.className = "raven-badge";
  badge.textContent = `Raven review ${Math.round(prediction.score * 100)}%`;

  const parent = node.parentElement;
  if (parent && !parent.querySelector(":scope > .raven-badge")) {
    parent.appendChild(badge);
  }

  return true;
}

let scanning = false;

async function scanPage() {
  if (scanning) return { scanned: 0, flagged: 0, source: "busy" };
  scanning = true;

  const candidates = getCandidates();
  let summary = { scanned: 0, flagged: 0, source: "none" };
  if (candidates.length) {
    const predictions = await predictBatch(candidates.map((item) => item.text));
    let flagged = 0;
    candidates.forEach((item, index) => {
      if (markNode(item.node, predictions[index])) flagged += 1;
    });
    summary = {
      scanned: candidates.length,
      flagged,
      source: predictions[0]?.source || "raven-api",
      updatedAt: new Date().toISOString()
    };
    chrome.storage?.local?.set({ ravenLastScan: summary });
  }

  scanning = false;
  return summary;
}

const observer = new MutationObserver(() => {
  window.clearTimeout(window.__ravenScanTimer);
  window.__ravenScanTimer = window.setTimeout(scanPage, 600);
});

observer.observe(document.documentElement, { childList: true, subtree: true });
scanPage();

chrome.runtime?.onMessage?.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "RAVEN_SCAN_NOW") return false;

  scanPage()
    .then((summary) => sendResponse({ ok: true, summary }))
    .catch((error) => sendResponse({ ok: false, error: error.message }));

  return true;
});
