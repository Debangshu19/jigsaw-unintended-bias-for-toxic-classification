import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowLeft,
  ArrowUp,
  BrainCircuit,
  CheckCircle2,
  Menu,
  MessageSquare,
  Play,
  Plus,
  Puzzle,
  ScanLine,
  Server,
  Trash2,
  X
} from "lucide-react";
import "./styles.css";
import Grainient from "./Grainient";
import { explainText, prettyCategory, scoreText } from "./ravenClient";

const howItWorks = [
  {
    img: "/assets/raven-playground-card.png",
    title: "Open the playground",
    desc: "Type a post, press Enter, and see the Raven score without crowding the landing page."
  },
  {
    img: "/assets/raven-x-inline-card.png",
    title: "Scan the web feed",
    desc: "Raven reads visible post text and places a quiet score beside the content."
  },
  {
    img: "/assets/raven-review-queue-card.png",
    title: "Highlight risky posts",
    desc: "Safe posts stay calm. Toxic posts move into a clean review queue."
  }
];

const features = [
  {
    img: "/assets/raven-web-showcase.png",
    title: "Built around the web, not a phone mockup",
    desc: "Raven keeps safe comments quiet and highlights borderline or harmful text with confidence signals."
  },
  {
    img: "/assets/raven-fallback-card.png",
    title: "Local model first. Fallback ready.",
    desc: "Serve your model locally and keep an AI Gateway fallback available for outages."
  },
  {
    img: "/assets/raven-x-inline-card.png",
    title: "Designed for X/Twitter timelines",
    desc: "The extension integrates into social pages with inline score chips instead of a separate phone screen."
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

const chatSamples = [
  "You are absolutely stupid and everyone hates you.",
  "Great article, thanks for sharing this perspective!",
  "I disagree with your take, but I respect the effort.",
  "Shut up you worthless loser, nobody wants you here."
];

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
    <a className="brand" href="/" aria-label="Raven home">
      <span className="brand-mark" aria-hidden="true" />
      <span className={light ? "brand-name is-light" : "brand-name"}>raven</span>
    </a>
  );
}

const VERDICTS = {
  toxic: { title: "This is toxic", badge: "Toxic" },
  borderline: { title: "Looks borderline", badge: "Borderline" },
  safe: { title: "This is safe", badge: "Safe" }
};

function toneFor(result) {
  if (result.needsReview) return "toxic";
  if ((result.score || 0) >= 0.35) return "borderline";
  return "safe";
}

function ChatVerdict({ result }) {
  const tone = toneFor(result);
  const percent = Math.round((result.score || 0) * 100);
  const { title, badge } = VERDICTS[tone];

  const categories = result.categories || {};
  const topCats = Object.entries(categories)
    .filter(([key, value]) => key !== "toxic" && value >= 0.2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  const words = Array.isArray(result.words) ? result.words : [];
  const maxImp = words.reduce((max, word) => Math.max(max, word.s || 0), 0);
  const threshold = Math.max(0.05, maxImp * 0.22);
  const hasHighlights = tone !== "safe" && maxImp > 0.05;
  const showWhy = tone !== "safe" && (topCats.length > 0 || hasHighlights);

  return (
    <div className={`verdict tone-${tone}`}>
      <div className="verdict-top">
        <span className="verdict-logo" aria-hidden="true" />
        <strong className="verdict-title">{title}</strong>
        <span className="verdict-chip">
          {badge}
          <em>{percent}%</em>
        </span>
      </div>

      {showWhy && (
        <div className="verdict-why">
          {topCats.length > 0 && (
            <div className="verdict-cats">
              {topCats.map(([key, value]) => (
                <span className="cat-chip" key={key}>
                  {prettyCategory(key)} <i>{Math.round(value * 100)}%</i>
                </span>
              ))}
            </div>
          )}
          {hasHighlights && (
            <p className="verdict-text">
              {words.map((word, index) => (
                <span key={index} className={(word.s || 0) >= threshold ? "tox-word" : ""}>
                  {word.w}
                  {index < words.length - 1 ? " " : ""}
                </span>
              ))}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function loadConversations() {
  try {
    const raw = window.localStorage.getItem("ravenChats");
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function App() {
  const [route, setRoute] = useState(() => (window.location.pathname === "/playground" ? "playground" : "home"));
  const [navLight, setNavLight] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  const [conversations, setConversations] = useState(loadConversations);
  const [activeId, setActiveId] = useState(null);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);

  const composerRef = useRef(null);
  const threadRef = useRef(null);
  const idRef = useRef(0);

  const nextId = () => {
    idRef.current += 1;
    return idRef.current;
  };

  const activeChat = conversations.find((c) => c.id === activeId) || null;
  const messages = activeChat ? activeChat.messages : [];

  useEffect(() => {
    try {
      window.localStorage.setItem("ravenChats", JSON.stringify(conversations));
    } catch {
      /* storage unavailable */
    }
  }, [conversations]);

  const sendMessage = useCallback(
    async (raw) => {
      const text = (raw ?? input).trim();
      if (!text || pending) return;

      setInput("");
      if (composerRef.current) composerRef.current.style.height = "auto";

      const userMessage = { id: nextId(), role: "user", text };
      let convId = activeId;

      if (convId == null) {
        convId = `chat-${nextId()}`;
        const title = text.length > 42 ? `${text.slice(0, 42)}…` : text;
        setConversations((prev) => [{ id: convId, title, messages: [userMessage] }, ...prev]);
        setActiveId(convId);
      } else {
        setConversations((prev) =>
          prev.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, userMessage] } : c))
        );
      }

      setPending(true);

      let result;
      try {
        result = await explainText(text);
      } catch {
        result = { text, ...scoreText(text), categories: {}, top_category: null, words: [] };
      }

      const ravenMessage = { id: nextId(), role: "raven", text, result };
      setConversations((prev) =>
        prev.map((c) => (c.id === convId ? { ...c, messages: [...c.messages, ravenMessage] } : c))
      );
      setPending(false);
      window.setTimeout(() => composerRef.current?.focus(), 0);
    },
    [input, pending, activeId]
  );

  const newChat = useCallback(() => {
    setActiveId(null);
    setInput("");
    if (composerRef.current) composerRef.current.style.height = "auto";
    window.setTimeout(() => composerRef.current?.focus(), 0);
  }, []);

  const deleteChat = useCallback(
    (event, id) => {
      event.stopPropagation();
      setConversations((prev) => prev.filter((c) => c.id !== id));
      setActiveId((current) => (current === id ? null : current));
    },
    []
  );

  useEffect(() => {
    const node = threadRef.current;
    if (node) node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [activeId, messages.length, pending]);

  useEffect(() => {
    const onPopState = () => setRoute(window.location.pathname === "/playground" ? "playground" : "home");
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((path) => {
    setMenuOpen(false);
    if (window.location.pathname !== path) {
      window.history.pushState({}, "", path);
    }
    setRoute(path === "/playground" ? "playground" : "home");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  useEffect(() => {
    if (route !== "playground") return undefined;
    const node = composerRef.current;
    const timer = window.setTimeout(() => node?.focus(), 180);
    return () => window.clearTimeout(timer);
  }, [route]);

  useEffect(() => {
    const onScroll = () => {
      const section = document.getElementById("how-it-works");
      if (!section || route === "playground") {
        setNavLight(route === "playground");
        return;
      }
      setNavLight(section.getBoundingClientRect().top <= 68);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, [route]);

  const openPlayground = useCallback(
    (event) => {
      event?.preventDefault();
      navigate("/playground");
    },
    [navigate]
  );

  const onComposerKeyDown = useCallback(
    (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    },
    [sendMessage]
  );

  const navLinks = [
    ["#how-it-works", "How it works"],
    ["#features", "Features"],
    ["#extension", "Extension"]
  ];
  const visibleNavLinks = route === "home" ? navLinks : [
    ["/", "Home"],
    ["/#how-it-works", "How it works"],
    ["/#extension", "Extension"]
  ];

  const handleNavClick = useCallback(
    (event, href) => {
      if (!href.startsWith("/")) {
        setMenuOpen(false);
        return;
      }

      event.preventDefault();
      const [, hash] = href.split("#");
      navigate("/");
      if (hash) {
        window.setTimeout(() => document.getElementById(hash)?.scrollIntoView({ behavior: "smooth" }), 180);
      }
    },
    [navigate]
  );

  return (
    <div id="top" className="site-shell">
      <nav className={`nav ${navLight ? "nav-light" : ""} ${route === "playground" ? "nav-app" : ""}`}>
        <div className="nav-inner">
          <Logo light={navLight} />
          <div className="nav-links">
            {visibleNavLinks.map(([href, label]) => (
              <a key={href} href={href} onClick={(event) => handleNavClick(event, href)}>
                {label}
              </a>
            ))}
          </div>
          <div className="nav-actions">
            {route === "home" ? (
              <a className="gradient-btn nav-cta" href="/playground" onClick={openPlayground}>
                <Play size={18} />
                Playground
              </a>
            ) : (
              <a className="nav-ghost" href="/" onClick={(event) => { event.preventDefault(); navigate("/"); }}>
                <ArrowLeft size={16} />
                Home
              </a>
            )}
          </div>
          <button className="menu-button" type="button" aria-label="Menu" onClick={() => setMenuOpen(!menuOpen)}>
            {menuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </nav>

      {menuOpen && (
        <div className="mobile-menu">
          {visibleNavLinks.map(([href, label]) => (
            <a key={href} href={href} onClick={(event) => handleNavClick(event, href)}>
              {label}
            </a>
          ))}
          <a className="gradient-btn" href="/playground" onClick={openPlayground}>
            <ScanLine size={18} />
            Playground
          </a>
        </div>
      )}

      <main>
        {route === "home" && (
          <>
        <section className="hero">
          <div className="hero-copy">
            <Reveal className="hero-title-wrap">
              <h1>Finally know what needs review.</h1>
            </Reveal>
            <Reveal delay={0.15} className="hero-side">
              <p>
                Raven scans comments, detects risky language, and highlights the posts that need human attention.
              </p>
              <a className="gradient-btn hero-btn" href="/playground" onClick={openPlayground}>
                <ScanLine size={18} />
                Open Playground
              </a>
            </Reveal>
          </div>
          <Reveal delay={0.3}>
            <div className="hero-stage">
              <img src="/assets/raven-hero-loop.gif" alt="Raven web and extension moderation preview" />
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
          </>
        )}

        {route === "playground" && (
          <section className="chat-page">
            <aside className="chat-sidebar">
              <button type="button" className="sidebar-new" onClick={newChat}>
                <Plus size={17} strokeWidth={2.6} />
                New chat
              </button>

              <div className="sidebar-history">
                <span className="sidebar-label">Recent chats</span>
                {conversations.length === 0 && <p className="sidebar-empty">No chats yet</p>}
                {conversations.map((chat) => (
                  <button
                    type="button"
                    key={chat.id}
                    className={`sidebar-item ${chat.id === activeId ? "is-active" : ""}`}
                    onClick={() => setActiveId(chat.id)}
                  >
                    <MessageSquare size={15} />
                    <span className="sidebar-item-title">{chat.title}</span>
                    <span className="sidebar-item-del" onClick={(event) => deleteChat(event, chat.id)} aria-label="Delete chat">
                      <Trash2 size={14} />
                    </span>
                  </button>
                ))}
              </div>
            </aside>

            <div className="chat-main">
              <div className="chat-bg" aria-hidden="true">
                <Grainient
                  color1="#ffffff"
                  color2="#43A5FF"
                  color3="#ffffff"
                  timeSpeed={0.25}
                  colorBalance={0}
                  warpStrength={1}
                  warpFrequency={5}
                  warpSpeed={2}
                  warpAmplitude={50}
                  blendAngle={0}
                  blendSoftness={0.05}
                  rotationAmount={500}
                  noiseScale={2}
                  grainAmount={0.1}
                  grainScale={2}
                  grainAnimated={false}
                  contrast={1.5}
                  gamma={1}
                  saturation={1}
                  centerX={0}
                  centerY={0}
                  zoom={0.9}
                />
              </div>

              <div className="chat-scroll" ref={threadRef}>
                {messages.length === 0 ? (
                  <div className="chat-hero">
                    <h1>What should Raven check?</h1>
                    <p>Paste any comment or post and Raven tells you instantly if it&apos;s toxic, borderline, or safe.</p>
                    <div className="chat-suggestions">
                      {chatSamples.map((sample) => (
                        <button type="button" key={sample} onClick={() => sendMessage(sample)}>
                          <span>{sample}</span>
                          <ArrowUp size={15} />
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="chat-messages">
                    {messages.map((message) =>
                      message.role === "user" ? (
                        <div className="chat-row is-user" key={message.id}>
                          <div className="chat-bubble">{message.text}</div>
                        </div>
                      ) : (
                        <div className="chat-row is-raven" key={message.id}>
                          <ChatVerdict result={message.result} />
                        </div>
                      )
                    )}
                    {pending && (
                      <div className="chat-row is-raven">
                        <div className="raven-loader">
                          <span className="loader-dots">
                            <i />
                            <i />
                            <i />
                          </span>
                          <span className="loader-shimmer">Raven is analyzing</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="chat-dock">
                <form
                  className="chat-composer"
                  onSubmit={(event) => {
                    event.preventDefault();
                    sendMessage();
                  }}
                >
                  <div className="prompt-input">
                    <textarea
                      className="prompt-input-field"
                      ref={composerRef}
                      rows={1}
                      value={input}
                      onChange={(event) => {
                        setInput(event.target.value);
                        const el = event.target;
                        el.style.height = "auto";
                        el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
                      }}
                      onKeyDown={onComposerKeyDown}
                      placeholder="Paste a comment to check…"
                      aria-label="Type a comment for Raven to check"
                    />
                    <div className="prompt-input-actions">
                      <span className="prompt-input-hint">Raven · DistilBERT toxicity model</span>
                      <button
                        type="submit"
                        className="prompt-send"
                        disabled={pending || !input.trim()}
                        aria-label="Send"
                      >
                        <ArrowUp size={18} strokeWidth={2.6} />
                      </button>
                    </div>
                  </div>
                </form>
              </div>
            </div>
          </section>
        )}

        {route === "home" && (
          <>
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
            <a className="gradient-btn" href="/playground" onClick={openPlayground}>
              <CheckCircle2 size={18} />
              Open Playground
            </a>
          </Reveal>
        </section>
          </>
        )}
      </main>

      {route === "home" && (
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
                  <a href="/playground" onClick={openPlayground}>Playground</a>
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
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
