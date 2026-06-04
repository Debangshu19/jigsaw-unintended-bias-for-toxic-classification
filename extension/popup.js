const apiState = document.getElementById("api-state");
const scanned = document.getElementById("scanned");
const flagged = document.getElementById("flagged");
const source = document.getElementById("source");
const scan = document.getElementById("scan");

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

scan.addEventListener("click", scanPage);
checkApi();
loadLastScan();
