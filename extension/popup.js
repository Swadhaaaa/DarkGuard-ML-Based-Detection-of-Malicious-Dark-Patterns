// ── Pattern metadata ──────────────────────────────────────
const PATTERNS = {
  fake_urgency:      { label: "Fake Urgency",     icon: "⏱" },
  hidden_cost:       { label: "Hidden Costs",     icon: "💸" },
  sneak_into_basket: { label: "Sneak To Basket",  icon: "🛒" },
  roach_motel:       { label: "Roach Motel",      icon: "🏨" },
  scarcity:          { label: "Scarcity",          icon: "📦" },
  privacy_zuckering: { label: "Privacy Zucking",  icon: "🕵️" },
  confusing_text:    { label: "Confusing Text",   icon: "💬" },
  trick_questions:   { label: "Trick Questions",  icon: "❓" },
  forced_action:     { label: "Forced Action",    icon: "🔒" },
  misdirection:      { label: "Misdirection",     icon: "👆" },
  confirmshaming:    { label: "Confirmshaming",   icon: "🥺" },
  social_proof_fake: { label: "Fake Social",  icon: "👥" },
  visual_trick:      { label: "Visual Tricks",    icon: "👁" },
};

// ── Helpers ───────────────────────────────────────────────
function show(id)  { document.getElementById(id).style.display = "block"; }
function hide(id)  { document.getElementById(id).style.display = "none";  }

function setApiStatus(state) {
  const el = document.getElementById("api-badge");
  if (!el) return;
  el.className = `badge-api ${state}`;
  el.textContent = state === "online" ? "ML API Online"
                 : state === "offline" ? "Local Fallback"
                 : "Connecting...";
}

function renderResults(data) {
  hide("view-loading");
  hide("view-error");
  show("view-results");

  const isDark = data.is_dark_pattern;

  // API badge
  setApiStatus(data.source === "ML_API" ? "online" : "offline");

  // Verdict card
  const card  = document.getElementById("verdict-card");
  card.className = `verdict-card ${isDark ? "dark" : "clean"}`;
  document.getElementById("v-icon").textContent  = isDark ? "🔴" : "🟢";
  document.getElementById("v-label").textContent = isDark ? "Dark Pattern Detected" : "Analysis Complete";
  document.getElementById("v-title").textContent = isDark ? "Malicious Intent Found" : "Interface Secured";
  document.getElementById("v-desc").textContent  = isDark
    ? "This page uses manipulative design patterns to influence your behaviour."
    : "No significant dark patterns detected. This site appears transparent.";

  // Score
  const score  = data.weighted_score ?? data.confidence ?? 0;
  const numEl  = document.getElementById("score-num");
  const fillEl = document.getElementById("score-fill");
  numEl.textContent  = score.toFixed(2);
  numEl.className    = `score-num ${isDark ? "dark" : "clean"}`;
  fillEl.className   = `fill ${isDark ? "dark" : "clean"}`;
  setTimeout(() => { fillEl.style.width = `${Math.min(score * 100, 100)}%`; }, 60);

  // Threat badge
  const threatEl = document.getElementById("threat-badge");
  const level    = data.threat_level || (isDark ? "HIGH" : "NONE");
  threatEl.textContent = level;
  threatEl.className   = `threat-badge threat-${level}`;
  document.getElementById("confidence-lbl").textContent =
    `Confidence: ${((data.confidence ?? 0) * 100).toFixed(1)}%`;

  // Pattern grid
  const grid   = document.getElementById("pattern-grid");
  const params = data.raw_params || {};
  grid.innerHTML = "";
  for (const [key, meta] of Object.entries(PATTERNS)) {
    const detected = (params[key] === 1) ||
      (Array.isArray(data.detected_patterns) && data.detected_patterns.includes(key));
    const pill = document.createElement("div");
    pill.className = `pattern-pill${detected ? " active" : ""}`;
    pill.innerHTML = `
      <div class="pill-dot"></div>
      <span class="pill-name">${meta.icon} ${meta.label}</span>
      <span class="pill-status">${detected ? "Found" : "None"}</span>`;
    grid.appendChild(pill);
  }

  // Source
  const srcEl = document.getElementById("source-badge");
  srcEl.textContent = data.source === "ML_API" ? "Ensemble ML (RF+GB+LR)" : "Local Heuristics";
  srcEl.className   = `source-badge ${data.source === "ML_API" ? "source-ml" : "source-local"}`;
}

function showError(msg) {
  hide("view-loading");
  hide("view-results");
  show("view-error");
  const errEl = document.getElementById("error-msg");
  if (errEl) errEl.textContent = msg;
  setApiStatus("offline");
}

// ── Scan logic ────────────────────────────────────────────
function runScan() {
  const btn = document.getElementById("btn-scan");
  if (btn) btn.disabled = true;
  
  show("view-loading");
  hide("view-results");
  hide("view-error");
  setApiStatus("loading");

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (!tabs || !tabs[0]) {
      showError("No active tab found.");
      if (btn) btn.disabled = false;
      return;
    }

    // Rely on manifest-injected content script
    chrome.tabs.sendMessage(tabs[0].id, { action: "scanPage" }, (response) => {
      if (btn) btn.disabled = false;

      if (chrome.runtime.lastError) {
        showError("Cannot scan this page.\nEnsure it's a website and try refreshing.");
        console.error("Scan error:", chrome.runtime.lastError.message);
        return;
      }
      
      if (!response) {
        showError("Received no response from page.");
        return;
      }

      if (response.error) {
        showError(response.error);
        return;
      }

      renderResults(response);
    });
  });
}

document.addEventListener('DOMContentLoaded', () => {
    const btnScan = document.getElementById("btn-scan");
    if (btnScan) {
        btnScan.addEventListener("click", runScan);
    }
    runScan();
});
