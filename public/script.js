// GeneScope front-end: talks to the FastAPI backend (/api/analyze, /api/health)
(() => {
    "use strict";

    // When served by FastAPI, use same-origin. If index.html is opened as a file, point at local server.
    const API_BASE = location.protocol === "file:" ? "http://localhost:8000" : "";
    const VIEW_LIMIT = 3000; // max bases drawn in the colored viewers (full text is still copied)
    const BASE_COLORS = { A: "var(--a)", T: "var(--t)", G: "var(--g)", C: "var(--c)" };

    const SAMPLE =
        "ATGGCGTACGCTAGCTAGGATCCGATCGATTACGGCGCGATATCGCGTAGCTAGCTAGGCTAATCGGATCGTAGCTAGCATCGATCGGCTAGCTAGGCCGATATCGCGCGTATATGCGCTAGCTAAGCTT";

    const $ = (id) => document.getElementById(id);
    const el = {
        seq: $("sequence"), count: $("char-count"), msg: $("message"),
        analyze: $("analyze-btn"), clear: $("clear-btn"),
        sample: $("sample-btn"), random: $("random-btn"), file: $("file-input"),
        results: $("results"), download: $("download-btn"),
        status: $("model-status"), statusText: $("model-status-text"),
    };

    let lastResult = null;

    /* ---------- helpers ---------- */
    function cleanSequence(raw) {
        return raw
            .split(/\r?\n/)
            .filter((l) => !l.trim().startsWith(">"))
            .join("")
            .replace(/\s+/g, "")
            .toUpperCase();
    }

    function showMessage(text, type = "error") {
        el.msg.textContent = text;
        el.msg.className = `message ${type}`;
        el.msg.hidden = false;
    }
    function hideMessage() { el.msg.hidden = true; }

    function setLoading(on) {
        el.analyze.disabled = on;
        el.analyze.classList.toggle("loading", on);
        el.analyze.querySelector(".btn-label").textContent = on ? "Analyzing…" : "Analyze DNA";
    }

    const fmtPct = (n) => `${n.toFixed(1)}%`;

    // Only A/C/G/T/N reach here (validated), so building HTML spans is safe.
    function colorize(seq, limit = VIEW_LIMIT) {
        const shown = seq.slice(0, limit);
        let html = "";
        for (const ch of shown) html += `<span class="${ch}">${ch}</span>`;
        if (seq.length > limit) html += `<span class="N">… (+${(seq.length - limit).toLocaleString()} more)</span>`;
        return html;
    }

    /* ---------- backend status ---------- */
    async function checkHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();
            if (data.model_loaded) {
                const acc = data.model_info?.test_accuracy;
                el.status.dataset.state = "ready";
                el.statusText.textContent = acc != null
                    ? `Model ready · ${(acc * 100).toFixed(1)}% acc`
                    : "Model ready";
            } else {
                el.status.dataset.state = "no-model";
                el.statusText.textContent = "Server online · model not trained";
            }
        } catch {
            el.status.dataset.state = "offline";
            el.statusText.textContent = "Server offline";
        }
    }

    /* ---------- input handling ---------- */
    function updateCount() {
        const n = cleanSequence(el.seq.value).length;
        el.count.textContent = `${n.toLocaleString()} base${n === 1 ? "" : "s"}`;
    }

    el.seq.addEventListener("input", () => { updateCount(); hideMessage(); });
    el.seq.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); analyze(); }
    });

    el.sample.addEventListener("click", () => {
        el.seq.value = SAMPLE; updateCount(); hideMessage();
    });

    el.random.addEventListener("click", () => {
        const bases = "ATGC";
        let s = "";
        for (let i = 0; i < 150; i++) s += bases[Math.floor(Math.random() * 4)];
        el.seq.value = s; updateCount(); hideMessage();
    });

    el.file.addEventListener("change", () => {
        const f = el.file.files[0];
        if (!f) return;
        if (f.size > 2 * 1024 * 1024) { showMessage("File is too large (max 2 MB)."); el.file.value = ""; return; }
        const reader = new FileReader();
        reader.onload = () => { el.seq.value = String(reader.result); updateCount(); hideMessage(); };
        reader.onerror = () => showMessage("Could not read that file.");
        reader.readAsText(f);
        el.file.value = "";
    });

    el.clear.addEventListener("click", () => {
        el.seq.value = ""; updateCount(); hideMessage();
        el.results.hidden = true; lastResult = null;
        el.seq.focus();
    });

    /* ---------- analyze ---------- */
    function parseError(data, status) {
        if (!data) return `Request failed (${status}).`;
        if (typeof data.detail === "string") return data.detail;
        if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join("; ");
        return `Request failed (${status}).`;
    }

    async function analyze() {
        hideMessage();
        const seq = cleanSequence(el.seq.value);

        if (!seq) return showMessage("Please enter a DNA sequence first.");
        const bad = [...new Set(seq.replace(/[ACGTN]/g, ""))];
        if (bad.length) {
            return showMessage(`Invalid character(s): ${bad.slice(0, 8).join(", ")}. Use only A, T, G, C.`);
        }

        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/api/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sequence: seq }),
            });
            let data = null;
            try { data = await res.json(); } catch { /* non-JSON error */ }
            if (!res.ok) throw new Error(parseError(data, res.status));

            lastResult = { ...data, sequence: seq };
            render(lastResult);
            el.results.hidden = false;
            el.results.scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (err) {
            if (err instanceof TypeError) {
                showMessage("Can't reach the server. Is it running? Start it with: uvicorn main:app --reload");
                checkHealth();
            } else {
                showMessage(err.message);
            }
        } finally {
            setLoading(false);
        }
    }
    el.analyze.addEventListener("click", analyze);

    /* ---------- render ---------- */
    function render(r) {
        const len = r.length;
        const c = r.counts;
        const pct = (n) => (len ? (n / len) * 100 : 0);

        $("length-result").textContent = len.toLocaleString();
        $("donut-length").textContent = len.toLocaleString();
        $("gc-result").textContent = fmtPct(r.gc_content);
        $("at-result").textContent = fmtPct(Math.max(0, 100 - r.gc_content - pct(r.n_count)));

        $("gc-fill").style.width = "0%";
        requestAnimationFrame(() => requestAnimationFrame(() => {
            $("gc-fill").style.width = `${Math.min(100, r.gc_content)}%`;
        }));
        $("gc-note").textContent =
            r.gc_content >= 60 ? "High GC content" :
            r.gc_content <= 40 ? "Low GC content" : "Moderate GC content";

        for (const b of ["A", "T", "G", "C"]) {
            $(`${b.toLowerCase()}-result`).textContent = c[b].toLocaleString();
            $(`${b.toLowerCase()}-pct`).textContent = `${fmtPct(pct(c[b]))} of sequence`;
        }

        renderDonut(c, len);
        renderPrediction(r.prediction);

        $("viewer-result").innerHTML = colorize(r.sequence);
        $("viewer-note").textContent = len > VIEW_LIMIT ? `Showing first ${VIEW_LIMIT.toLocaleString()} bases` : "";
        $("complement-result").innerHTML = colorize(r.complement);
        $("reverse-result").innerHTML = colorize(r.reverse_complement);
    }

    function renderDonut(counts, len) {
        const svg = $("donut");
        const legend = $("legend");
        svg.replaceChildren();
        legend.replaceChildren();

        const NS = "http://www.w3.org/2000/svg";
        const R = 15.9155; // circumference = 100
        const mk = (offset, dash, color, cls) => {
            const c = document.createElementNS(NS, "circle");
            c.setAttribute("cx", "21"); c.setAttribute("cy", "21"); c.setAttribute("r", R);
            if (cls) c.setAttribute("class", cls);
            if (color) c.setAttribute("stroke", color);
            c.setAttribute("stroke-dasharray", dash);
            c.setAttribute("stroke-dashoffset", offset);
            return c;
        };
        svg.appendChild(mk(0, "100 0", null, "track"));

        let acc = 0;
        for (const b of ["A", "T", "G", "C"]) {
            const p = len ? (counts[b] / len) * 100 : 0;
            const seg = mk(-acc, "0 100", BASE_COLORS[b]);
            svg.appendChild(seg);
            requestAnimationFrame(() => requestAnimationFrame(() => {
                seg.setAttribute("stroke-dasharray", `${Math.max(0, p - 0.4)} ${100 - Math.max(0, p - 0.4)}`);
            }));
            acc += p;

            const li = document.createElement("li");
            const dot = document.createElement("i");
            dot.style.background = BASE_COLORS[b];
            const name = document.createTextNode(b);
            const val = document.createElement("span");
            val.textContent = fmtPct(p);
            li.append(dot, name, val);
            legend.appendChild(li);
        }
    }

    function renderPrediction(pred) {
        const body = $("prediction-body");
        const empty = $("pred-empty");
        const list = $("prob-list");
        list.replaceChildren();

        if (!pred) {
            body.hidden = true; empty.hidden = false;
            return;
        }
        body.hidden = false; empty.hidden = true;

        $("pred-label").textContent = pred.label;
        $("pred-conf").textContent = fmtPct(pred.confidence * 100);
        $("pred-tag").hidden = pred.confidence >= 0.5;

        const entries = Object.entries(pred.probabilities).slice(0, 6);
        for (const [name, p] of entries) {
            const row = document.createElement("div");
            row.className = "prob-row";

            const n = document.createElement("span");
            n.className = "prob-name"; n.textContent = name; n.title = name;

            const track = document.createElement("div");
            track.className = "prob-track";
            const fill = document.createElement("div");
            fill.className = "prob-fill";
            track.appendChild(fill);

            const v = document.createElement("span");
            v.className = "prob-val"; v.textContent = fmtPct(p * 100);

            row.append(n, track, v);
            list.appendChild(row);
            requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${p * 100}%`; }));
        }
    }

    /* ---------- copy / download ---------- */
    async function copyText(text) {
        try {
            await navigator.clipboard.writeText(text);
        } catch {
            const ta = document.createElement("textarea");
            ta.value = text; document.body.appendChild(ta); ta.select();
            document.execCommand("copy"); ta.remove();
        }
    }

    document.querySelectorAll(".copy-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            if (!lastResult) return;
            await copyText(lastResult[btn.dataset.copy] || "");
            const old = btn.textContent;
            btn.textContent = "Copied ✓"; btn.classList.add("copied");
            setTimeout(() => { btn.textContent = old; btn.classList.remove("copied"); }, 1500);
        });
    });

    el.download.addEventListener("click", () => {
        if (!lastResult) return;
        const blob = new Blob([JSON.stringify(lastResult, null, 2)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "genescope-analysis.json";
        a.click();
        URL.revokeObjectURL(a.href);
    });

    /* ---------- init ---------- */
    updateCount();
    checkHealth();
})();
