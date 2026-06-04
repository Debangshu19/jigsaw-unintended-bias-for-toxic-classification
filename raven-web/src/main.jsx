import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowDownToLine,
  BrainCircuit,
  CheckCircle2,
  ExternalLink,
  Menu,
  Puzzle,
  RefreshCw,
  ScanLine,
  Send,
  Server,
  ShieldCheck,
  X
} from "lucide-react";
import "./styles.css";
import { checkRavenHealth, formatSourceName, parseDemoLines, scoreText, scoreWithApi } from "./ravenClient";

const howItWorks = [
  {
    img: "/assets/raven-onboarding.svg",
    title: "Load Raven",
    desc: "Use the Raven model service or connect your fine-tuned DistilBERT checkpoint."
  },
  {
    img: "/assets/raven-scan.svg",
    title: "Scan the page",
    desc: "Raven reads visible comments and scores what needs human review."
  },
  {
    img: "/assets/raven-extension.svg",
    title: "Highlight risky posts",
    desc: "The extension marks concerning comments in place without changing the original page."
  }
];

const features = [
  {
    img: "/assets/raven-scan.svg",
    title: "See exactly what needs review",
    desc: "Raven keeps safe comments quiet and highlights borderline or harmful text with confidence signals."
  },
  {
    img: "/assets/raven-dashboard.svg",
    title: "Use a model you can actually own",
    desc: "Start with the DistilBERT notebook work already in this repo, export it, and serve it as Raven."
  },
  {
    img: "/assets/raven-extension.svg",
    title: "Built for social feeds",
    desc: "The browser extension prototype scans visible comment areas on pages like video, post, and thread views."
  }
];

const pipeline = [
  {
    icon: BrainCircuit,
    title: "Train",
    label: "raven-model",
    desc: "Fine-tune DistilBERT from Jigsaw-style CSV data and export a Hugging Face checkpoint."
  },
  {
    icon: Server,
    title: "Serve",
    label: "raven-api",
    desc: "Run FastAPI on port 8000 and expose health, single prediction, and batch prediction endpoints."
  },
  {
    icon: Puzzle,
    title: "Highlight",
    label: "extension",
    desc: "Scan visible comments in the browser and mark the exact nodes Raven sends to review."
  }
];

const demoSamples = [
  "Great article. Thanks for sharing this perspective.",
  "That was not cool. Let's keep the discussion respectful.",
  "I disagree with the point, but the explanation helped.",
  "This sounds aggressive and should be reviewed by a moderator."
];

const quickSample = "This comment is aggressive and should be reviewed.";

function Reveal({ children, delay = 0, className = "" }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node) return undefined;
    const fallback = window.setTimeout(() => setVisible(true), 500);

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          window.clearTimeout(fallback);
          setVisible(true);
          observer.unobserve(node);
        }
      },
      { threshold: 0.16 }
    );

    observer.observe(node);
    return () => {
      window.clearTimeout(fallback);
      observer.disconnect();
    };
  }, []);

  return (
    <div
      ref={ref}
      className={`reveal ${visible ? "is-visible" : ""} ${className}`}
      style={{ transitionDelay: `${delay}s` }}
    >
      {children}
    </div>
  );
}

function Logo({ light = false }) {
  return (
    <a className="brand" href="#top" aria-label="Raven home">
      <span className="brand-mark">R</span>
      <span className={light ? "brand-name is-light" : "brand-name"}>raven</span>
    </a>
  );
}

function App() {
  const [navLight, setNavLight] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [demoText, setDemoText] = useState(demoSamples.join("\n"));
  const [scanNonce, setScanNonce] = useState(0);
  const [demoResults, setDemoResults] = useState(() =>
    demoSamples.map((line) => ({ text: line, ...scoreText(line) }))
  );
  const [demoSource, setDemoSource] = useState("browser-demo-fallback");
  const [demoLoading, setDemoLoading] = useState(false);
  const [apiHealth, setApiHealth] = useState({ state: "checking", source: "checking" });
  const [quickText, setQuickText] = useState(quickSample);
  const [quickResult, setQuickResult] = useState(() => ({ text: quickSample, ...scoreText(quickSample) }));
  const [quickLoading, setQuickLoading] = useState(false);
  const quickInputRef = useRef(null);

  const demoLines = useMemo(() => parseDemoLines(demoText), [demoText]);

  const checkApiHealth = useCallback(async () => {
    setApiHealth((current) => ({ ...current, state: "checking" }));

    try {
      const health = await checkRavenHealth();
      setApiHealth({ state: "online", source: health.source || "raven-api" });
      return health;
    } catch {
      setApiHealth({ state: "offline", source: "browser fallback" });
      return null;
    }
  }, []);

  useEffect(() => {
    checkApiHealth();
  }, [checkApiHealth]);

  useEffect(() => {
    const controller = new AbortController();

    if (!demoLines.length) {
      setDemoResults([]);
      setDemoSource("empty");
      setDemoLoading(false);
      return () => controller.abort();
    }

    setDemoLoading(true);
    const timer = window.setTimeout(async () => {
      try {
        const apiResults = await scoreWithApi(demoLines, controller.signal);
        setDemoResults(apiResults);
        setDemoSource(apiResults[0]?.source || "raven-api");
        setApiHealth({ state: "online", source: apiResults[0]?.source || "raven-api" });
      } catch (error) {
        if (controller.signal.aborted) return;
        setDemoResults(demoLines.map((line) => ({ text: line, ...scoreText(line) })));
        setDemoSource("browser-demo-fallback");
        setApiHealth({ state: "offline", source: "browser fallback" });
      } finally {
        if (!controller.signal.aborted) setDemoLoading(false);
      }
    }, 320);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [demoLines, scanNonce]);

  useEffect(() => {
    const onScroll = () => {
      const section = document.getElementById("how-it-works");
      if (!section) return;
      setNavLight(section.getBoundingClientRect().top <= 68);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const focusDemo = useCallback((event) => {
    event?.preventDefault();
    setMenuOpen(false);
    document.getElementById("demo")?.scrollIntoView({ behavior: "smooth", block: "start" });
    window.setTimeout(() => quickInputRef.current?.focus(), 520);
  }, []);

  const runQuickScan = useCallback(
    async (event) => {
      event?.preventDefault();
      const line = quickText.trim();

      if (!line) {
        setQuickResult(null);
        return;
      }

      setQuickLoading(true);
      try {
        const [apiResult] = await scoreWithApi([line]);
        setQuickResult(apiResult);
        setApiHealth({ state: "online", source: apiResult?.source || "raven-api" });
      } catch {
        setQuickResult({ text: line, ...scoreText(line) });
        setApiHealth({ state: "offline", source: "browser fallback" });
      } finally {
        setQuickLoading(false);
      }
    },
    [quickText]
  );

  const navLinks = [
    ["#how-it-works", "How it works"],
    ["#features", "Features"],
    ["#engine", "Engine"]
  ];

  return (
    <div id="top" className="site-shell">
      <nav className={`nav ${navLight ? "nav-light" : ""}`}>
        <div className="nav-inner">
          <Logo light={navLight} />
          <div className="nav-links">
            {navLinks.map(([href, label]) => (
              <a key={href} href={href}>
                {label}
              </a>
            ))}
          </div>
          <div className="nav-actions">
            <span className={`api-pill ${apiHealth.state}`}>{apiHealth.state === "online" ? "API online" : apiHealth.state === "checking" ? "Checking" : "Fallback"}</span>
            <a className="gradient-btn nav-cta" href="#demo" onClick={focusDemo}>
              <ArrowDownToLine size={18} />
              Try Demo
            </a>
          </div>
          <button className="menu-button" type="button" aria-label="Menu" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </nav>

      {menuOpen && (
        <div className="mobile-menu">
          {navLinks.map(([href, label]) => (
            <a key={href} href={href} onClick={() => setMenuOpen(false)}>
              {label}
            </a>
          ))}
          <a className="gradient-btn" href="#demo" onClick={focusDemo}>
            <ScanLine size={18} />
            Try Demo
          </a>
        </div>
      )}

      <main>
        <section className="hero">
          <div className="hero-copy">
            <Reveal className="hero-title-wrap">
              <h1>Finally know what needs review.</h1>
            </Reveal>
            <Reveal delay={0.15} className="hero-side">
              <p>
                Raven scans comments, detects risky language, and highlights the posts that need human attention.
              </p>
              <a className="gradient-btn hero-btn" href="#demo" onClick={focusDemo}>
                <ScanLine size={18} />
                Scan Sample Text
              </a>
              <div className="hero-signals" aria-label="Raven system status">
                <span>
                  <strong>{apiHealth.state === "online" ? "Live" : "Demo"}</strong>
                  API path
                </span>
                <span>
                  <strong>Batch</strong>
                  comment scan
                </span>
                <span>
                  <strong>MV3</strong>
                  extension
                </span>
              </div>
            </Reveal>
          </div>
          <Reveal delay={0.3}>
            <div className="hero-stage">
              <img src="/assets/raven-hero-stage.svg" alt="Raven mobile app scanning comments" />
            </div>
          </Reveal>
        </section>

        <section id="how-it-works" className="light-section">
          <div className="section-inner">
            <Reveal>
              <div className="section-head">
                <span>How it works</span>
                <h2>Up and running in 2 minutes.</h2>
                <p>Connect the model service, scan visible comments, and review only the posts Raven flags.</p>
              </div>
            </Reveal>

            <div className="step-grid">
              {howItWorks.map((item, index) => (
                <Reveal key={item.title} delay={index * 0.1}>
                  <article className="step-card">
                    <img src={item.img} alt={item.title} />
                    <h3>{item.title}</h3>
                    <p>{item.desc}</p>
                  </article>
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        <section id="features" className="light-section features-section">
          <div className="section-inner">
            <Reveal>
              <div className="section-head">
                <span>Features</span>
                <h2>Simple tools. Real results.</h2>
                <p>No complicated setup. No fake claims. Raven can run from your own fine-tuned model.</p>
              </div>
            </Reveal>

            <div className="feature-list">
              {features.map((item, index) => (
                <div className={`feature-row ${index % 2 ? "is-reversed" : ""}`} key={item.title}>
                  <Reveal className="feature-media">
                    <img src={item.img} alt={item.title} />
                  </Reveal>
                  <Reveal delay={0.15} className="feature-copy">
                    <h3>{item.title}</h3>
                    <p>{item.desc}</p>
                  </Reveal>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="demo" className="light-section demo-section">
          <div className="section-inner">
            <Reveal>
              <div className="section-head">
                <span>Raven Lab</span>
                <h2>Try the scanner.</h2>
                <p>This demo calls `raven-api` first. If the service is offline, it falls back to the local browser demo scorer.</p>
              </div>
            </Reveal>

            <Reveal>
              <form className="quick-demo" onSubmit={runQuickScan}>
                <div className="quick-demo-copy">
                  <span>Live check</span>
                  <strong>Type a comment and press Enter.</strong>
                </div>
                <div className="quick-input-row">
                  <input
                    id="quick-demo-input"
                    ref={quickInputRef}
                    value={quickText}
                    onChange={(event) => setQuickText(event.target.value)}
                    placeholder="Write a comment to scan"
                    aria-label="Type a comment to scan"
                  />
                  <button type="submit" disabled={quickLoading}>
                    <Send size={18} />
                    {quickLoading ? "Scanning" : "Scan"}
                  </button>
                </div>
                <div
                  className={`quick-result ${quickResult?.needsReview ? "needs-review" : quickResult ? "is-safe" : ""}`}
                  aria-live="polite"
                >
                  {quickResult ? (
                    <>
                      <strong>{quickResult.needsReview ? "Review" : "Safe"}</strong>
                      <span>{Math.round(quickResult.score * 100)}% · {formatSourceName(quickResult.source)}</span>
                    </>
                  ) : (
                    <span>Enter a comment to scan it.</span>
                  )}
                </div>
              </form>
            </Reveal>

            <div className="demo-grid">
              <Reveal>
                <div className="demo-editor">
                  <textarea
                    value={demoText}
                    onChange={(event) => setDemoText(event.target.value)}
                    aria-label="Sample comments to scan"
                  />
                  <div className="demo-actions">
                    <button type="button" onClick={() => setScanNonce((value) => value + 1)}>
                      <RefreshCw size={14} />
                      Scan now
                    </button>
                    <button type="button" onClick={() => setDemoText(demoSamples.join("\n"))}>
                      Reset
                    </button>
                    <button type="button" onClick={checkApiHealth}>
                      Check API
                    </button>
                    <a href="#engine">
                      Model path <ExternalLink size={14} />
                    </a>
                  </div>
                </div>
              </Reveal>

              <Reveal delay={0.15}>
                <div className="result-panel">
                  <div className="result-title">
                    <span>
                      <ShieldCheck size={20} />
                      Raven Review Queue
                    </span>
                    <em className={demoSource === "browser-demo-fallback" ? "source-fallback" : ""}>
                      {demoLoading ? "Scanning..." : formatSourceName(demoSource)}
                    </em>
                  </div>
                  {demoResults.map((result, index) => (
                    <div className={`result-item ${result.needsReview ? "needs-review" : ""}`} key={`${result.text}-${index}`}>
                      <p>{result.text}</p>
                      <span>
                        {result.needsReview ? "Review" : "Safe"} · {Math.round(result.score * 100)}%
                      </span>
                    </div>
                  ))}
                  {!demoResults.length && <p className="empty-state">Add a comment to scan.</p>}
                </div>
              </Reveal>
            </div>
          </div>
        </section>

        <section id="engine" className="light-section engine-section">
          <div className="section-inner">
            <Reveal>
              <div className="section-head">
                <span>Engine</span>
                <h2>One model path. Three surfaces.</h2>
                <p>Raven is wired so the same classifier can power the website demo, the API, and the browser extension.</p>
              </div>
            </Reveal>
            <div className="pipeline-grid">
              {pipeline.map((item, index) => {
                const Icon = item.icon;
                return (
                  <Reveal key={item.title} delay={index * 0.1}>
                    <article className="pipeline-card">
                      <div className="pipeline-icon">
                        <Icon size={22} />
                      </div>
                      <span>{item.label}</span>
                      <h3>{item.title}</h3>
                      <p>{item.desc}</p>
                    </article>
                  </Reveal>
                );
              })}
            </div>
          </div>
        </section>

        <section id="extension" className="light-section extension-band">
          <Reveal>
            <h2>Bring Raven to the browser.</h2>
            <p>The extension prototype scans visible comment nodes and highlights anything the model marks for review.</p>
            <a className="gradient-btn" href="#demo" onClick={focusDemo}>
              <CheckCircle2 size={18} />
              Test the Flow
            </a>
          </Reveal>
        </section>
      </main>

      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-top">
            <div>
              <Logo />
              <p>AI-assisted comment moderation for safer online spaces.</p>
            </div>
            <div className="footer-links">
              <div>
                <h4>Product</h4>
                <a href="#how-it-works">How it works</a>
                <a href="#features">Features</a>
                <a href="#demo">Demo</a>
              </div>
              <div>
                <h4>Build</h4>
                <a href="#features">Plan</a>
                <a href="#extension">Extension</a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>© {new Date().getFullYear()} Raven. All rights reserved.</p>
            <p>Built for local model ownership.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
