export const RAVEN_API_URL = import.meta.env.VITE_RAVEN_API_URL || "http://localhost:8000";

export const reviewTerms = [
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

export function parseDemoLines(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

export function scoreText(text) {
  const normalized = text.toLowerCase();
  const hits = reviewTerms.filter((term) => normalized.includes(term)).length;
  const score = Math.min(0.94, 0.14 + hits * 0.26 + Math.min(text.length / 420, 0.18));
  const needsReview = hits > 0 || score > 0.58;

  return {
    score,
    needsReview,
    needs_review: needsReview,
    label: needsReview ? "review" : "safe",
    source: "browser-demo-fallback"
  };
}

export function formatSourceName(source) {
  if (!source) return "Raven engine";
  if (source === "raven-hf-model" || source === "raven-local-model") return "Raven engine";
  if (source.includes("fallback")) return "Demo fallback";
  if (source === "raven-api") return "Raven engine";
  return source;
}

export async function checkRavenHealth() {
  const response = await fetch(`${RAVEN_API_URL}/health`);
  if (!response.ok) throw new Error(`Raven API returned ${response.status}`);
  return response.json();
}

export async function scoreWithApi(lines, signal) {
  const response = await fetch(`${RAVEN_API_URL}/predict-batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: lines }),
    signal
  });

  if (!response.ok) {
    throw new Error(`Raven API returned ${response.status}`);
  }

  const data = await response.json();
  if (!Array.isArray(data.predictions)) {
    throw new Error("Raven API response is missing predictions");
  }

  return data.predictions.map((prediction, index) => ({
    text: lines[index],
    score: Number(prediction.score) || 0,
    needsReview: Boolean(prediction.needs_review),
    needs_review: Boolean(prediction.needs_review),
    label: prediction.label || (prediction.needs_review ? "review" : "safe"),
    source: prediction.source || "raven-api"
  }));
}
