// Service worker: runs the model call here so the page's CSP/CORS can never block it.
const RAVEN_API = "http://localhost:8000/predict-batch";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "RAVEN_PREDICT") return false;

  fetch(RAVEN_API, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: message.texts })
  })
    .then((response) => {
      if (!response.ok) throw new Error(`Raven API ${response.status}`);
      return response.json();
    })
    .then((data) => {
      if (!Array.isArray(data.predictions)) throw new Error("bad payload");
      sendResponse({ ok: true, predictions: data.predictions });
    })
    .catch((error) => sendResponse({ ok: false, error: String(error.message || error) }));

  return true; // keep the message channel open for the async response
});
