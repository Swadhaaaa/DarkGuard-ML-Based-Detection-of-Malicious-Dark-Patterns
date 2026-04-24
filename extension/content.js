// ============================================================
//  DarkGuard ML — Content Script v2.1
//  Scans the current webpage for dark patterns, then sends
//  signals to the background which calls the ML API server.
//  Falls back to LOCAL heuristic scoring if API is unavailable.
// ============================================================

const DARKGUARD_CONFIG = {
    API_URL:   "http://127.0.0.1:5000/predict",
    TIMEOUT_MS: 3000,    // 3s timeout before local fallback kicks in
    THRESHOLD:  0.50,
    WEIGHTS: {
        fake_urgency:      0.12,
        hidden_cost:       0.12,
        sneak_into_basket: 0.12,
        roach_motel:       0.10,
        scarcity:          0.08,
        privacy_zuckering: 0.08,
        confusing_text:    0.08,
        trick_questions:   0.08,
        forced_action:     0.06,
        misdirection:      0.05,
        confirmshaming:    0.05,
        social_proof_fake: 0.04,
        visual_trick:      0.02,
    }
};

// ─────────────────────────────────────────────────────────
// DETECTION ENGINE  (heuristic signal extraction from DOM)
// ─────────────────────────────────────────────────────────

const Signals = {

    fake_urgency: (doc) => {
        const patterns = [
            /\b(hurry|limited time|ends (today|tonight|soon|in \d+)|flash sale|act now|last chance|don.t miss|expires? (today|soon|in)|offer ends|deal ends|selling out)\b/i,
            /\b(only \d+ (hours?|mins?|minutes?|days?) left|time is running out)\b/i,
            /\d{1,2}:\d{2}:\d{2}/,          // countdown timer HH:MM:SS
            /\d{1,2}h\s*\d{1,2}m\s*\d{1,2}s/i,  // 2h 30m 10s
        ];
        const text = doc.body?.innerText || "";
        if (patterns.some(p => p.test(text))) return 1;

        // Detect live countdown elements (numeric elements ticking)
        const timerEls = doc.querySelectorAll('[class*="countdown"],[class*="timer"],[id*="countdown"],[id*="timer"]');
        if (timerEls.length > 0) return 1;

        return 0;
    },

    scarcity: (doc) => {
        const patterns = [
            /\b(only \d+ (left|remaining|in stock)|just \d+ left|low stock|almost gone|selling fast|in high demand|limited stock|last \d+ (items?|units?))\b/i,
            /\b(\d+\s*(people|viewers?|shoppers?|customers?)\s*(are\s*)?(viewing|watching|looking at|have this))\b/i,
            /\b(nearly sold out|almost sold out|few left|going fast)\b/i,
        ];
        return patterns.some(p => p.test(doc.body?.innerText || "")) ? 1 : 0;
    },

    confusing_text: (doc) => {
        const patterns = [
            /opt.?out.{0,60}(uncheck|untick)/i,
            /do\s+not\s+uncheck/i,
            /by (continuing|clicking|proceeding|using).{0,80}you (agree|consent|accept)/i,
            /no,?\s+i\s+(don.t\s+want|hate\s+(discounts?|savings?|deals?))/i,
            /unsubscribe from (all|promotional|marketing|our)\s+email/i,
            /keep\s+my\s+(regular|higher|full)\s+price/i,
            /i\s+prefer\s+to\s+pay\s+(more|full\s+price)/i,
        ];
        return patterns.some(p => p.test(doc.body?.innerText || "")) ? 1 : 0;
    },

    hidden_cost: (doc) => {
        const patterns = [
            /\b(service fee|booking fee|resort fee|convenience fee|processing fee|handling fee|platform fee|admin fee)\b/i,
            /\b(taxes?\s*(and\s*fees?)?\s*(not\s*)?(included|apply|extra)|plus\s+tax|excl\.\s+tax|excluding\s+vat)\b/i,
            /\b(additional (charges?|fees?|costs?) (may\s+)?apply)\b/i,
            /\b(shipping\s+(not\s+)?included|delivery\s+fee\s+applies)\b/i,
        ];
        return patterns.some(p => p.test(doc.body?.innerText || "")) ? 1 : 0;
    },

    forced_action: (doc) => {
        const patterns = [
            /\b(must (sign.?up|register|create (an\s+)?account|log.?in)|account required|no guest (checkout|option))\b/i,
            /\b(sign.?up to (continue|proceed|view|access|download|get))\b/i,
            /\b(registration (is\s+)?mandatory|login required to (continue|checkout))\b/i,
        ];
        if (patterns.some(p => p.test(doc.body?.innerText || ""))) return 1;

        // Detect if checkout/continue buttons are walled behind a login modal
        const modals = doc.querySelectorAll('[class*="modal"],[class*="dialog"],[role="dialog"]');
        for (const m of modals) {
            if (/sign.?up|log.?in|create account/i.test(m.innerText || "")) return 1;
        }
        return 0;
    },

    social_proof_fake: (doc) => {
        const patterns = [
            /\b(\d{3,},?\d*\s+(people|customers?|users?|members?|shoppers?)\s+(love|trust|use|joined|bought|purchased|rated))\b/i,
            /\b(trusted by (millions|thousands|hundreds of thousands))\b/i,
            /\b(verified (buyer|purchase|review))\b/i,
            /\b(award.?winning|#1\s+(rated|recommended|trusted|selling))\b/i,
            /\b(\d+ (people|customers?) (recently|just) (bought|purchased|viewed))\b/i,
        ];
        return patterns.some(p => p.test(doc.body?.innerText || "")) ? 1 : 0;
    },

    misdirection: (doc) => {
        const negativeLabels = [
            /\bno,?\s+i\s+(don.t|hate|dislike|prefer not|won.t)\b/i,
            /\b(no thanks?|i don.t want|skip (the\s+)?(deal|offer|discount|savings?)|decline|not interested)\b/i,
            /\bi prefer to pay (more|full price|higher price)\b/i,
            /\b(not? right now|maybe later)\b/i,
        ];
        const clickables = doc.querySelectorAll('button, a[href], input[type="button"], input[type="submit"], [role="button"]');
        for (const el of clickables) {
            const label = (el.innerText || el.value || el.getAttribute("aria-label") || "").trim();
            if (negativeLabels.some(p => p.test(label))) return 1;
        }
        return 0;
    },

    visual_trick: (doc) => {
        // 1. Tiny text (< 8px on meaningful content)
        const textEls = doc.querySelectorAll("p, span, label, small, div");
        for (const el of textEls) {
            if (!el.innerText || el.innerText.trim().length < 5) continue;
            const style = window.getComputedStyle(el);
            const fontSize = parseFloat(style.fontSize);
            if (fontSize > 0 && fontSize < 8) return 1;
        }

        // 2. Low contrast cancel / decline button (grey on grey)
        const actionEls = doc.querySelectorAll("button, a, input[type='submit']");
        for (const el of actionEls) {
            const label = (el.innerText || el.value || "").toLowerCase();
            if (!/(cancel|close|decline|skip|no thanks)/i.test(label)) continue;
            const style = window.getComputedStyle(el);
            const color  = style.color;
            const bg     = style.backgroundColor;
            // Identical color & background = invisible text
            if (color === bg && !["transparent", "rgba(0, 0, 0, 0)"].includes(bg)) return 1;
        }

        // 3. Pre-checked opt-in checkboxes
        const checkboxes = doc.querySelectorAll('input[type="checkbox"]');
        for (const cb of checkboxes) {
            if (!cb.checked) continue;
            const labelEl = doc.querySelector(`label[for="${cb.id}"]`);
            const labelText = labelEl?.innerText || cb.closest("label")?.innerText || "";
            if (/(newsletter|promo|marketing|subscription|offer|deal)/i.test(labelText)) return 1;
        }

        return 0;
    },

    confirmshaming: (doc) => {
        const patterns = [
            /\b(no,? i( don.?t| cannot| won.?t) want (savings|discounts|to save|cash\s*back|offers|deals))\b/i,
            /\b(actually\s+not\s+want\s+(automatic|savings|cash\s*back))\b/i,
            /\b(i prefer to pay (more|full price))\b/i,
            /\b(i.?m not interested in (improving|saving|learning))\b/i,
        ];
        return patterns.some(p => p.test(doc.body?.innerText || "")) ? 1 : 0;
    },

    sneak_into_basket: (doc) => {
        const patterns = [
            /\b(added to (your )?(cart|basket) (automatically|by default))\b/i,
            /\b(bonus|extra) item included\b/i,
        ];
        if (patterns.some(p => p.test(doc.body?.innerText || ""))) return 1;
        
        // Look for pre-checked "add-on" or "insurance" checkboxes in checkout context
        const isCheckout = /\b(checkout|cart|basket|order)\b/i.test(doc.title + (doc.location?.href || ""));
        if (!isCheckout) return 0;
        
        const checkboxes = doc.querySelectorAll('input[type="checkbox"]');
        for (const cb of checkboxes) {
            if (!cb.checked) continue;
            const text = (cb.closest("label")?.innerText || "").toLowerCase();
            if (/(insurance|protection|coverage|warranty|add-on|donation)/.test(text)) return 1;
        }
        return 0;
    },

    roach_motel: (doc) => {
        const text = doc.body?.innerText || "";
        const patterns = [
            /\b(to cancel( your subscription| this service)?, please call( us at)?)\b/i,
            /\b(cancellations (must be|are) done (over the phone|by calling))\b/i,
        ];
        return patterns.some(p => p.test(text)) ? 1 : 0;
    },

    privacy_zuckering: (doc) => {
        // Detecting forced data sharing or convoluted privacy settings
        const patterns = [
            /\b(we share (your )?data with (our )?partners)\b/i,
            /\b(by using this (site|app|service), you agree to (all )?our data sharing)\b/i,
            /\b(train (our )?ai on your (content|data))\b/i,
        ];
        return patterns.some(p => p.test(doc.body?.innerText || "")) ? 1 : 0;
    },

    trick_questions: (doc) => {
        const text = doc.body?.innerText || "";
        const patterns = [
            /\b(uncheck to opt\s*in)\b/i,
            /\b(check to (unsubscribe|opt\s*out))\b/i,
            /\b(do you (actually\s+)?not want)\b/i,
            /\b(please do not send me)\b/i,
        ];
        return patterns.some(p => p.test(text)) ? 1 : 0;
    },
};

// ─────────────────────────────────────────────────────────
// LOCAL FALLBACK scoring (used if API is unreachable)
// ─────────────────────────────────────────────────────────

function localScore(params) {
    let score = 0;
    for (const [key, weight] of Object.entries(DARKGUARD_CONFIG.WEIGHTS)) {
        score += (params[key] || 0) * weight;
    }
    return parseFloat(score.toFixed(4));
}

function buildLocalResult(params) {
    const score       = localScore(params);
    const patternCount = Object.values(params).reduce((a, b) => a + b, 0);
    
    // User requested override: If any pattern is found, consider it a dark pattern
    const isDark      = (score >= DARKGUARD_CONFIG.THRESHOLD) || (patternCount >= 1);
    const confidence  = Math.min(0.55 + score * 0.4, 0.99);  // approximate

    let threat;
    if (!isDark)                    threat = "NONE";
    else if (score < 0.30)          threat = "LOW";  // Guaranteed minimum LOW if forced to dark
    else if (score < 0.50)          threat = "LOW";
    else if (score < 0.70)          threat = "MEDIUM";
    else if (score < 0.85)          threat = "HIGH";
    else                            threat = "CRITICAL";

    return {
        is_dark_pattern:   isDark,
        confidence:        parseFloat(confidence.toFixed(4)),
        threat_level:      threat,
        weighted_score:    score,
        pattern_count:     patternCount,
        detected_patterns: Object.keys(params).filter(k => params[k] === 1),
        certainty:         "LOCAL_FALLBACK",  // flag that this is heuristic-only
        version:           "local-heuristic",
    };
}

// ─────────────────────────────────────────────────────────
// MAIN SCAN FUNCTION
// ─────────────────────────────────────────────────────────

async function scanPage() {
    console.log("[DarkGuard] Starting page analysis...");

    // Extract raw signals from DOM
    const params = {};
    for (const [key, detector] of Object.entries(Signals)) {
        try {
            params[key] = detector(document);
        } catch (e) {
            console.warn(`[DarkGuard] Signal '${key}' failed:`, e);
            params[key] = 0;
        }
    }

    console.log("[DarkGuard] Raw signals:", params);

    // Try ML API first (with timeout)
    try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), DARKGUARD_CONFIG.TIMEOUT_MS);

        const response = await fetch(DARKGUARD_CONFIG.API_URL, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                Fake_Urgency:      params.fake_urgency,
                Hidden_Cost:       params.hidden_cost,
                Sneak_Into_Basket: params.sneak_into_basket,
                Roach_Motel:       params.roach_motel,
                Scarcity:          params.scarcity,
                Privacy_Zuckering: params.privacy_zuckering,
                Confusing_Text:    params.confusing_text,
                Trick_Questions:   params.trick_questions,
                Forced_Action:     params.forced_action,
                Misdirection:      params.misdirection,
                Confirmshaming:    params.confirmshaming,
                Social_Proof_Fake: params.social_proof_fake,
                Visual_Trick:      params.visual_trick,
            }),
            signal: controller.signal,
        });
        clearTimeout(timer);

        if (response.ok) {
            const json = await response.json();
            const result = { ...json.data, raw_params: params, source: "ML_API" };
            console.log("[DarkGuard] ML API result:", result);
            return result;
        }
    } catch (e) {
        if (e.name === "AbortError") {
            console.warn("[DarkGuard] API timeout — using local fallback.");
        } else {
            console.warn("[DarkGuard] API unreachable — using local fallback.", e.message);
        }
    }

    // Local fallback
    const result = { ...buildLocalResult(params), raw_params: params, source: "LOCAL_FALLBACK" };
    console.log("[DarkGuard] Local fallback result:", result);
    return result;
}

// ─────────────────────────────────────────────────────────
// MESSAGE LISTENER  (popup → content script)
// ─────────────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "scanPage") {
        scanPage().then(result => sendResponse(result)).catch(err => {
            console.error("[DarkGuard] Scan error:", err);
            sendResponse({ error: err.message });
        });
        return true;  // keeps channel open for async response
    }
});
