import { useMemo, useState } from "react";
import {
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { Ionicons } from "@expo/vector-icons";

const API_URL = process.env.EXPO_PUBLIC_RAVEN_API_URL || "http://127.0.0.1:8000";

const initialText = [
  "Great article. Thanks for sharing this perspective.",
  "That was not cool. Let's keep the discussion respectful.",
  "I disagree with the point, but the explanation helped.",
  "This sounds aggressive and should be reviewed by a moderator."
].join("\n");

const quickSample = "This comment is aggressive and should be reviewed.";

function fallbackScore(text) {
  const terms = ["aggressive", "attack", "bully", "hate", "hurt", "insult", "moderator", "not cool", "review", "stupid", "threat", "harass"];
  const normalized = text.toLowerCase();
  const hits = terms.filter((term) => normalized.includes(term)).length;
  const score = Math.min(0.96, 0.12 + hits * 0.28 + Math.min(text.length / 500, 0.18));
  const needsReview = hits > 0 || score >= 0.5;
  return {
    label: needsReview ? "review" : "safe",
    score,
    needs_review: needsReview,
    source: "mobile-demo-fallback"
  };
}

function linesFromText(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean);
}

function formatSource(sourceName) {
  if (!sourceName) return "Raven engine";
  if (sourceName === "raven-hf-model" || sourceName === "raven-local-model" || sourceName === "raven-api") {
    return "Raven engine";
  }
  if (sourceName === "raven-ai-gateway-fallback") return "Raven fallback";
  if (sourceName.includes("fallback")) return "Demo fallback";
  return sourceName;
}

export default function App() {
  const [text, setText] = useState(initialText);
  const [results, setResults] = useState(linesFromText(initialText).map((line) => ({ text: line, ...fallbackScore(line) })));
  const [source, setSource] = useState("mobile-demo-fallback");
  const [loading, setLoading] = useState(false);
  const [quickText, setQuickText] = useState(quickSample);
  const [quickResult, setQuickResult] = useState({ text: quickSample, ...fallbackScore(quickSample) });
  const [quickLoading, setQuickLoading] = useState(false);

  const reviewCount = useMemo(() => results.filter((result) => result.needs_review).length, [results]);

  async function predictLines(lines) {
    try {
      const response = await fetch(`${API_URL}/predict-batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines })
      });
      if (!response.ok) throw new Error("API offline");
      const data = await response.json();
      const predictions = data.predictions.map((prediction, index) => ({
        text: lines[index],
        ...prediction
      }));
      return predictions;
    } catch {
      return lines.map((line) => ({ text: line, ...fallbackScore(line) }));
    }
  }

  async function scanQuick() {
    const line = quickText.trim();
    if (!line) {
      setQuickResult(null);
      return;
    }

    setQuickLoading(true);
    const [prediction] = await predictLines([line]);
    setQuickResult(prediction);
    setSource(prediction?.source || "raven-api");
    setQuickLoading(false);
  }

  async function scan() {
    const lines = linesFromText(text);
    if (!lines.length) {
      setResults([]);
      return;
    }

    setLoading(true);
    try {
      const predictions = await predictLines(lines);
      setResults(predictions);
      setSource(predictions[0]?.source || "raven-api");
    } finally {
      setLoading(false);
    }
  }

  return (
    <SafeAreaView style={styles.shell}>
      <StatusBar style="dark" />
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <View style={styles.mark}><Text style={styles.markText}>R</Text></View>
          <View>
            <Text style={styles.title}>Raven</Text>
            <Text style={styles.subtitle}>Comment review queue</Text>
          </View>
        </View>

        <View style={styles.heroCard}>
          <Ionicons name="shield-checkmark-outline" size={54} color="#258dff" />
          <Text style={styles.heroTitle}>Scan comments</Text>
          <Text style={styles.heroCopy}>Find posts that need moderator attention before they spread.</Text>
          <View style={styles.stats}>
            <View><Text style={styles.statValue}>{results.length}</Text><Text style={styles.statLabel}>Scanned</Text></View>
            <View><Text style={styles.statValue}>{reviewCount}</Text><Text style={styles.statLabel}>Review</Text></View>
          </View>
        </View>

        <View style={styles.quickCard}>
          <Text style={styles.quickLabel}>Live check</Text>
          <TextInput
            style={styles.quickInput}
            value={quickText}
            onChangeText={setQuickText}
            placeholder="Type a comment and press Enter"
            returnKeyType="go"
            onSubmitEditing={scanQuick}
          />
          <Pressable style={styles.quickButton} onPress={scanQuick} disabled={quickLoading}>
            <Text style={styles.quickButtonText}>{quickLoading ? "Scanning..." : "Scan comment"}</Text>
          </Pressable>
          <Text style={[styles.quickResult, quickResult?.needs_review && styles.quickReview]}>
            {quickResult
              ? `${quickResult.needs_review ? "Review" : "Safe"} · ${Math.round(quickResult.score * 100)}% · ${formatSource(quickResult.source)}`
              : "Enter a comment to scan it"}
          </Text>
        </View>

        <TextInput
          style={styles.input}
          multiline
          value={text}
          onChangeText={setText}
          placeholder="Paste comments, one per line"
        />

        <Pressable style={styles.button} onPress={scan} disabled={loading}>
          <Text style={styles.buttonText}>{loading ? "Scanning..." : "Scan now"}</Text>
        </Pressable>
        <Text style={styles.source}>Source: {formatSource(source)}</Text>

        <View style={styles.results}>
          {results.map((result, index) => (
            <View key={`${result.text}-${index}`} style={[styles.result, result.needs_review && styles.review]}>
              <Text style={styles.resultText}>{result.text}</Text>
              <Text style={[styles.badge, result.needs_review && styles.reviewBadge]}>
                {result.needs_review ? "Review" : "Safe"} · {Math.round(result.score * 100)}%
              </Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  shell: { flex: 1, backgroundColor: "#ffffff" },
  content: { padding: 22, gap: 18 },
  header: { flexDirection: "row", alignItems: "center", gap: 12 },
  mark: { width: 42, height: 42, borderRadius: 12, backgroundColor: "#a9dfff", alignItems: "center", justifyContent: "center" },
  markText: { fontSize: 18, fontWeight: "800", color: "#122033" },
  title: { fontSize: 24, fontWeight: "800", color: "#111" },
  subtitle: { color: "#7b8794", fontWeight: "600" },
  heroCard: { padding: 24, borderRadius: 24, backgroundColor: "#f0f8ff", alignItems: "center", gap: 10 },
  heroTitle: { fontSize: 30, fontWeight: "800", color: "#111" },
  heroCopy: { textAlign: "center", color: "#6c7785", lineHeight: 21 },
  stats: { width: "100%", flexDirection: "row", gap: 12, marginTop: 10 },
  statValue: { fontSize: 30, fontWeight: "800", color: "#111" },
  statLabel: { color: "#7b8794", fontWeight: "800", textTransform: "uppercase", fontSize: 12 },
  quickCard: { padding: 16, borderRadius: 20, backgroundColor: "#fff", borderWidth: 1, borderColor: "#e1eef8", gap: 10 },
  quickLabel: { color: "#258dff", fontWeight: "800", textTransform: "uppercase", fontSize: 12 },
  quickInput: { height: 48, paddingHorizontal: 14, borderRadius: 14, backgroundColor: "#f7fbff", color: "#111" },
  quickButton: { height: 46, borderRadius: 23, backgroundColor: "#258dff", alignItems: "center", justifyContent: "center" },
  quickButtonText: { color: "#fff", fontWeight: "800" },
  quickResult: { color: "#208044", fontWeight: "800" },
  quickReview: { color: "#a45d00" },
  input: { minHeight: 150, padding: 16, borderRadius: 18, backgroundColor: "#f7fbff", color: "#111", textAlignVertical: "top" },
  button: { height: 54, borderRadius: 27, backgroundColor: "#258dff", alignItems: "center", justifyContent: "center" },
  buttonText: { color: "#fff", fontSize: 17, fontWeight: "800" },
  source: { textAlign: "center", color: "#7b8794", fontWeight: "700" },
  results: { gap: 12 },
  result: { padding: 16, borderRadius: 16, borderWidth: 1, borderColor: "#dcefe2", backgroundColor: "#fff" },
  review: { borderColor: "#f2a323", backgroundColor: "#fffaf0" },
  resultText: { color: "#243244", lineHeight: 20, marginBottom: 10 },
  badge: { color: "#208044", fontWeight: "800" },
  reviewBadge: { color: "#a45d00" }
});
