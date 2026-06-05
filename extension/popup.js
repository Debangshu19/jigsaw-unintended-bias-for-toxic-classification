const API_BASE = "http://localhost:8000";

const apiState = document.getElementById("api-state");
const scanned = document.getElementById("scanned");
const flagged = document.getElementById("flagged");
const rescan = document.getElementById("rescan");

function formatSource(name) {
  if (!name) return "Raven engine";
  if (name === "raven-hf-model" || name === "raven-local-model" || name === "raven-api") return "Raven engine";
  if (name === "raven-ai-gateway-fallback") return "Raven fallback";
  if (name.includes("fallback")) return "demo fallback";
  return name;
}

function renderSummary(s) {
  scanned.textContent = s?.scanned ?? 0;
  flagged.textContent = s?.flagged ?? 0;
}

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error("offline");
    const data = await response.json();
    apiState.textContent = `Engine online · ${formatSource(data.source)}`;
  } catch {
    apiState.textContent = "Engine offline · demo fallback";
  }
}

async function loadLastScan() {
  try {
    const result = await chrome.storage.local.get("ravenLastScan");
    renderSummary(result.ravenLastScan);
  } catch {
    renderSummary(null);
  }
}

async function rescanPage() {
  rescan.disabled = true;
  rescan.textContent = "Scanning…";
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const response = await chrome.tabs.sendMessage(tab.id, { type: "RAVEN_SCAN_NOW" });
    if (response?.ok) renderSummary(response.summary);
  } catch {
    apiState.textContent = "Open & reload an X / YouTube tab";
  } finally {
    rescan.disabled = false;
    rescan.textContent = "Rescan now";
  }
}

rescan.addEventListener("click", rescanPage);
checkApi();
loadLastScan();
