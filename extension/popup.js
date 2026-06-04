const apiState = document.getElementById("api-state");
const scanned = document.getElementById("scanned");
const flagged = document.getElementById("flagged");
const source = document.getElementById("source");
const scan = document.getElementById("scan");
const quickForm = document.getElementById("quick-form");
const quickText = document.getElementById("quick-text");
const quickScan = document.getElementById("quick-scan");
const quickResult = document.getElementById("quick-result");

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

function renderSummary(summary) {
  scanned.textContent = summary?.scanned ?? 0;
  flagged.textContent = summary?.flagged ?? 0;
  source.textContent = summary?.source ? `Source: ${summary.source}` : "No scan yet";
}

async function checkApi() {
  try {
    const response = await fetch("http://localhost:8000/health");
    if (!response.ok) throw new Error("offline");
    const data = await response.json();
    apiState.textContent = `API online: ${data.source}`;
  } catch {
    apiState.textContent = "API offline: using fallback";
  }
}

function fallbackPredict(text) {
  const normalized = text.toLowerCase();
  const hits = fallbackTerms.filter((term) => normalized.includes(term)).length;
  const score = Math.min(0.95, 0.12 + hits * 0.3 + Math.min(text.length / 500, 0.16));
  const needsReview = hits > 0 || score >= 0.58;

  return {
    label: needsReview ? "review" : "safe",
    needs_review: needsReview,
    score,
    source: "extension-demo-fallback"
  };
}

async function predictText(text) {
  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!response.ok) throw new Error("Raven API unavailable");
    return response.json();
  } catch {
    return fallbackPredict(text);
  }
}

function renderQuickResult(prediction) {
  quickResult.className = `quick-result ${prediction.needs_review ? "review" : "safe"}`;
  quickResult.textContent = `${prediction.needs_review ? "Review" : "Safe"} · ${Math.round(prediction.score * 100)}% · ${prediction.source}`;
}

async function loadLastScan() {
  const result = await chrome.storage.local.get("ravenLastScan");
  renderSummary(result.ravenLastScan);
}

async function scanPage() {
  scan.disabled = true;
  scan.textContent = "Scanning...";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const response = await chrome.tabs.sendMessage(tab.id, { type: "RAVEN_SCAN_NOW" });
    if (!response?.ok) throw new Error(response?.error || "No Raven content script on this page");
    renderSummary(response.summary);
  } catch (error) {
    source.textContent = error.message;
  } finally {
    scan.disabled = false;
    scan.textContent = "Scan this page";
  }
}

quickForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = quickText.value.trim();
  if (!text) {
    quickResult.className = "quick-result";
    quickResult.textContent = "Type a comment first";
    quickText.focus();
    return;
  }

  quickScan.disabled = true;
  quickScan.textContent = "Checking";
  quickResult.className = "quick-result";
  quickResult.textContent = "Scanning...";

  const prediction = await predictText(text);
  renderQuickResult(prediction);

  quickScan.disabled = false;
  quickScan.textContent = "Check";
});

scan.addEventListener("click", scanPage);
checkApi();
loadLastScan();
