"use strict";

// ASD Gene Predict: a static viewer for the thesis rankings.
// Data: data/genes.json (one row per gene), data/meta.json, data/annotations/NN.json
// (UniProt/HGNC detail, loaded on demand; NN = last two digits of the Ensembl ID).

const PAGE = 100;
const HOT_KEYWORDS = new Set([
  "Autism", "Autism spectrum disorder", "Intellectual disability", "Epilepsy", "Schizophrenia",
  "Neurodegeneration", "Neurogenesis", "Synapse", "Chromatin regulator",
]);
const TYPE_LABEL = { pc: "protein-coding", nc: "non-coding RNA", ps: "pseudogene", o: "other" };
const VIEWS = {
  candidates: {
    test: (g) => g.role === "" || g.role === "new",
    note: "Genes with no label when the models were trained (January 2024). Orange marks the ones SFARI has added since.",
  },
  new: {
    test: (g) => g.role === "new",
    note: "Genes SFARI added between January 2024 and July 2026. None of them were labelled when the models were trained.",
  },
  known: {
    test: (g) => g.role === "train_pos" || g.role === "sfari_2024",
    note: "SFARI genes as of January 2024. Category 1 and syndromic genes were training positives; categories 2 and 3 were not used for training.",
  },
  all: {
    test: () => true,
    note: "Every gene with a score, including the training genes and the training negatives.",
  },
};
const SORTS = {
  protein: (a, b) => nullLast(a.pr, b.pr),
  graph: (a, b) => nullLast(a.gr, b.gr),
  mean: (a, b) => nullLast(a.mean, b.mean),
  symbol: (a, b) => a.sym.localeCompare(b.sym),
};

const state = {
  genes: [], bySymbol: new Map(), byId: new Map(), meta: null,
  view: "candidates", sort: "protein", query: "", shown: PAGE,
  rows: [], selected: null, shards: new Map(), known: { pr: [], gr: [] }, chart: "protein",
};
const $ = (id) => document.getElementById(id);

function nullLast(a, b) {
  if (a == null && b == null) return 0;
  if (a == null) return 1;
  if (b == null) return -1;
  return a - b;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
const int = (n) => Number(n).toLocaleString("en-US");
function topLabel(t) {
  if (t == null) return "–";
  if (t < 0.1) return "<0.1%";
  return (t < 10 ? t.toFixed(1) : Math.round(t)) + "%";
}
async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

// ------------------------------------------------------------------ data
async function load() {
  const [table, meta] = await Promise.all([getJSON("data/genes.json"), getJSON("data/meta.json")]);
  state.meta = meta;
  const cols = table.columns;
  state.genes = table.rows.map((row) => Object.fromEntries(cols.map((c, i) => [c, row[i]])));

  for (const g of state.genes) {
    g.ptop = g.pr != null ? (100 * g.pr) / meta.n_protein : null;
    g.gtop = g.gr != null ? (100 * g.gr) / meta.n_graph : null;
    const tops = [g.ptop, g.gtop].filter((t) => t != null);
    g.mean = tops.length ? tops.reduce((s, t) => s + t, 0) / tops.length : null;
    g.symU = g.sym.toUpperCase();
    g.altU = (g.alt || "").toUpperCase().split("|").filter(Boolean);
    g.nameL = (g.name || "").toLowerCase();
    state.byId.set(g.id, g);
    state.bySymbol.set(g.symU, g);
    if (g.role === "train_pos" || g.role === "sfari_2024") {
      if (g.ptop != null) state.known.pr.push(g.ptop);
      if (g.gtop != null) state.known.gr.push(g.gtop);
    }
  }
  // Rank among the genes unlabelled in 2024 (the temporal-validation pool).
  for (const key of ["pr", "gr"]) {
    const pool = state.genes.filter((g) => g.pool && g[key] != null).sort((a, b) => a[key] - b[key]);
    pool.forEach((g, i) => { g["c" + key] = i + 1; });
    state["pool_" + key] = pool.length;
  }
  state.density = { pr: density(state.known.pr), gr: density(state.known.gr) };
}

function shardOf(id) {
  const key = id.slice(-2);
  if (!state.shards.has(key)) {
    state.shards.set(key, getJSON(`data/annotations/${key}.json`).catch(() => ({})));
  }
  return state.shards.get(key);
}

// ------------------------------------------------------------------ list
function matches(g, q) {
  if (!q) return true;
  const Q = q.toUpperCase();
  if (g.symU.startsWith(Q) || g.id === Q) return true;
  if (g.altU.some((a) => a === Q || (Q.length > 2 && a.startsWith(Q)))) return true;
  return q.length > 2 && g.nameL.includes(q.toLowerCase());
}

function computeRows() {
  const q = state.query.trim();
  const view = VIEWS[state.view];
  let rows = state.genes.filter((g) => view.test(g) && matches(g, q));
  rows.sort(SORTS[state.sort]);
  if (q) {
    const Q = q.toUpperCase();
    rows.sort((a, b) => (b.symU === Q) - (a.symU === Q)); // exact symbol first, order kept otherwise
  }
  state.rows = rows;
}

function badge(g) {
  if (g.role === "new") return `<span class="badge new" title="Added to SFARI after training, category ${esc(g.s26)}">New · ${esc(g.s26)}</span>`;
  if (g.role === "train_pos" || g.role === "sfari_2024") {
    return `<span class="badge known" title="SFARI ${g.s24 === "S" ? "syndromic" : "category " + esc(g.s24)} in January 2024">SFARI ${esc(g.s24)}</span>`;
  }
  if (g.role === "train_neg") return `<span class="badge neg" title="Training negative (Krishnan et al. 2016)">Negative</span>`;
  return `<span class="badge none" aria-label="No label">–</span>`;
}

function miniTrack(top) {
  const pct = top == null ? 0 : 100 - top;
  return `<span class="t"><span class="bar"><i style="width:${pct.toFixed(1)}%"></i></span><span class="v">${topLabel(top)}</span></span>`;
}

function renderList() {
  const list = $("list");
  const rows = state.rows.slice(0, state.shown);
  list.innerHTML = rows.map((g, i) => `
    <li><button type="button" class="row" data-id="${g.id}" ${state.selected === g ? 'aria-current="true"' : ""}>
      <span class="pos">${int(i + 1)}</span>
      <span class="gene"><span class="sym">${esc(g.sym)}</span><span class="nm">${esc(g.name || "")}</span></span>
      <span class="mini" aria-label="Protein model top ${topLabel(g.ptop)}, network model top ${topLabel(g.gtop)}">${miniTrack(g.ptop)}${miniTrack(g.gtop)}</span>
      ${badge(g)}
    </button></li>`).join("");
  const more = $("more");
  const left = state.rows.length - rows.length;
  more.hidden = left <= 0;
  more.textContent = `Show ${int(Math.min(PAGE, left))} more of ${int(left)}`;
  $("empty").hidden = state.rows.length > 0;
  $("view-note").textContent = VIEWS[state.view].note + (state.query ? ` ${int(state.rows.length)} match “${state.query.trim()}”.` : "");
}

function refresh(resetPaging = true) {
  if (resetPaging) state.shown = PAGE;
  computeRows();
  renderList();
}

// ------------------------------------------------------------------ detail
// Share of the SFARI genes known in 2024 in each of 50 slices of a ranking (left = lowest
// scores, right = highest), scaled so the fullest slice is 1.
function density(tops, bins = 50) {
  const counts = new Array(bins).fill(0);
  for (const t of tops) counts[Math.min(bins - 1, Math.floor(((100 - t) / 100) * bins))]++;
  const max = Math.max(...counts, 1);
  return counts.map((c) => c / max);
}

function trackPlot(top, key) {
  const W = 1000, H = 30, base = 26;
  const dens = state.density[key];
  const bw = W / dens.length;
  const bars = dens.map((d, i) => d > 0
    ? `<rect x="${(i * bw + 1).toFixed(1)}" y="${(base - d * (base - 2)).toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${(d * (base - 2)).toFixed(1)}"/>`
    : "").join("");
  const marker = top == null ? "" : `<span class="marker" style="left:calc(${(100 - top).toFixed(2)}% - 1px)"></span>`;
  return `<div class="plot"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
      <g fill="var(--known)" fill-opacity="0.45">${bars}</g>
      <rect x="0" y="${base}" width="${W}" height="4" fill="var(--track)"/>
    </svg>${marker}</div>`;
}

function trackBlock(label, g, key) {
  const rank = g[key];
  const n = key === "pr" ? state.meta.n_protein : state.meta.n_graph;
  const top = key === "pr" ? g.ptop : g.gtop;
  const info = rank == null ? "not scored" : `rank ${int(rank)} of ${int(n)} · top ${topLabel(top)}`;
  return `<div class="track">
    <div class="track-label"><span>${label}</span><span class="rk">${info}</span></div>
    ${trackPlot(top, key)}
    <div class="axis"><span>lowest score</span><span>highest</span></div>
  </div>`;
}

function statusBlock(g) {
  const s26 = g.s26 == null ? null : g.s26 === "S" ? "syndromic, no score" : `category ${g.s26}`;
  const s24 = g.s24 === "S" ? "a syndromic gene without a score" : `category ${g.s24}`;
  const cand = g.cpr ? ` Among the ${int(state.pool_pr)} genes with no label in 2024, it ranked <strong>#${int(g.cpr)}</strong> in the protein model${g.cgr ? ` and #${int(g.cgr)} in the network model` : ""}.` : "";
  switch (g.role) {
    case "new":
      return `<p class="status new"><strong>Added to SFARI after the models were trained</strong> (${esc(s26)} in July 2026). It had no label in the training data.${cand}</p>`;
    case "train_pos": {
      const moved = g.s26 && g.s26 !== g.s24 ? ` In July 2026 it is ${esc(s26)}.` : "";
      return `<p class="status known"><strong>Training positive:</strong> SFARI ${esc(s24)} in January 2024. Its score comes from cross-validation, from a model that did not see it.${moved}</p>`;
    }
    case "sfari_2024": {
      const moved = g.s26 && g.s26 !== g.s24 ? ` SFARI moved it to ${esc(s26)} in July 2026.` : "";
      return `<p class="status known"><strong>SFARI ${esc(s24)}</strong> in January 2024. Not used for training, so its score is a prediction.${moved}</p>`;
    }
    case "train_neg": {
      const later = g.s26 ? ` SFARI has since added it (${esc(s26)}).` : "";
      return `<p class="status neg"><strong>Training negative</strong> from Krishnan et al. (2016). Its score comes from cross-validation.${later}</p>`;
    }
    default:
      return `<p class="status none"><strong>No label.</strong> Not in SFARI (July 2026) and not a training negative.${cand}</p>`;
  }
}

function linksBlock(g, ann) {
  const sym = encodeURIComponent(g.sym);
  const links = [];
  if (g.s24 || g.s26) links.push(["SFARI Gene", `https://gene.sfari.org/database/human-gene/${sym}`]);
  if (g.acc) links.push(["UniProt", `https://www.uniprot.org/uniprotkb/${g.acc}/entry`]);
  links.push(["Ensembl", `https://www.ensembl.org/Homo_sapiens/Gene/Summary?g=${g.id}`]);
  links.push(["gnomAD", `https://gnomad.broadinstitute.org/gene/${g.id}?dataset=gnomad_r4`]);
  if (ann && ann.omim) links.push(["OMIM", `https://omim.org/entry/${ann.omim}`]);
  links.push(["GeneCards", `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${sym}`]);
  return `<div class="links">${links.map(([t, u]) => `<a href="${u}" target="_blank" rel="noopener">${t}</a>`).join("")}</div>`;
}

function loeufBlock(g) {
  if (g.loeuf == null) return `<p class="muted">No gnomAD constraint estimate for this gene.</p>`;
  const pos = Math.min(g.loeuf, 2) / 2 * 100;
  const note = g.loeuf < 0.6
    ? "Below 0.6, the gnomAD v4 threshold for genes intolerant to loss-of-function variants. Many neurodevelopmental genes are."
    : "Above 0.6: the gene tolerates loss-of-function variants in the general population more readily.";
  return `<div class="loeuf-row"><span class="val">LOEUF ${g.loeuf.toFixed(2)}</span>
      <span class="loeuf-scale" role="img" aria-label="LOEUF ${g.loeuf.toFixed(2)} on a scale from 0 to 2"><i style="left:calc(${pos.toFixed(1)}% - 1px)"></i></span></div>
    <p class="loeuf-note">${note}</p>`;
}

function annotationBlocks(g, ann) {
  if (!ann) return `<section class="d-section"><h3>Function</h3><p class="muted">No UniProt or HGNC annotation for this gene.</p></section>`;
  const parts = [];
  const fn = ann.function
    ? `<p class="function">${esc(ann.function)}</p>${ann.more && g.acc ? `<a class="more-link" href="https://www.uniprot.org/uniprotkb/${g.acc}/entry#function" target="_blank" rel="noopener">Full description on UniProt</a>` : ""}`
    : `<p class="muted">UniProt has no function description for this gene.</p>`;
  parts.push(`<section class="d-section"><h3>Function${ann.protein ? ` · <span style="text-transform:none;letter-spacing:0">${esc(ann.protein)}</span>` : ""}</h3>${fn}
    ${ann.group ? `<p class="group">Gene group: ${esc(ann.group.split("|").join(", "))}</p>` : ""}</section>`);
  if (ann.diseases && ann.diseases.length) {
    parts.push(`<section class="d-section"><h3>Disease involvement (UniProt)</h3><ul class="diseases">${ann.diseases.map(([name, mim]) =>
      `<li>${esc(name)}${mim ? `<a class="mim" href="https://omim.org/entry/${mim}" target="_blank" rel="noopener">MIM ${mim}</a>` : ""}</li>`).join("")}</ul></section>`);
  }
  if (ann.keywords && ann.keywords.length) {
    const kw = [...ann.keywords].sort((a, b) => HOT_KEYWORDS.has(b) - HOT_KEYWORDS.has(a));
    parts.push(`<section class="d-section"><h3>Keywords</h3><div class="chips">${kw.map((k) =>
      `<span class="chip${HOT_KEYWORDS.has(k) ? " hot" : ""}">${esc(k)}</span>`).join("")}</div></section>`);
  }
  return parts.join("");
}

async function renderDetail(g) {
  const el = $("detail");
  const ids = [g.loc, TYPE_LABEL[g.type], g.id, g.acc].filter(Boolean).map((s) => `<span>${esc(s)}</span>`).join("");
  const alt = g.alt ? `<p class="d-ids"><span>Also: ${esc(g.alt.split("|").slice(0, 6).join(", "))}</span></p>` : "";
  const head = `
    <button type="button" class="close" id="close">← Back to the list</button>
    <div class="d-top"><h2 class="d-sym">${esc(g.sym)}</h2>${badge(g)}</div>
    <p class="d-name">${esc(g.name || "")}</p>
    <p class="d-ids">${ids}</p>${alt}
    ${statusBlock(g)}
    <section class="d-section"><h3>Where it ranks</h3>
      ${trackBlock("Protein model", g, "pr")}
      ${trackBlock("Network model", g, "gr")}
      <p class="track-note"><span class="key">where the SFARI genes known in 2024 fall</span><span class="key here">${esc(g.sym)}</span></p>
    </section>`;
  const tail = `
    <section class="d-section"><h3>Tolerance to loss of function (gnomAD)</h3>${loeufBlock(g)}</section>
    <section class="d-section"><h3>Elsewhere</h3><div id="links">${linksBlock(g, null)}</div></section>
    <p class="disclaimer">A research ranking from an MSc thesis, not a diagnostic tool.</p>`;
  el.innerHTML = head + `<div id="ann"><section class="d-section"><h3>Function</h3><p class="muted">Loading…</p></section></div>` + tail;
  $("close").addEventListener("click", closeDetail);

  const shard = await shardOf(g.id);
  if (state.selected !== g) return; // another gene was picked meanwhile
  const ann = shard[g.id];
  $("ann").innerHTML = annotationBlocks(g, ann);
  $("links").innerHTML = linksBlock(g, ann);
}

function select(g, { open = false, scroll = false } = {}) {
  if (!g) return;
  state.selected = g;
  document.querySelectorAll(".row[aria-current]").forEach((r) => r.removeAttribute("aria-current"));
  const row = document.querySelector(`.row[data-id="${g.id}"]`);
  if (row) {
    row.setAttribute("aria-current", "true");
    if (scroll) row.scrollIntoView({ block: "nearest" });
  }
  renderDetail(g);
  history.replaceState(null, "", "#" + encodeURIComponent(g.sym));
  const detail = $("detail");
  detail.scrollTop = 0;
  if (open && matchMedia("(max-width: 960px)").matches) {
    detail.classList.add("open");
    document.body.style.overflow = "hidden";
    detail.focus();
  }
}

function closeDetail() {
  $("detail").classList.remove("open");
  document.body.style.overflow = "";
  const row = state.selected && document.querySelector(`.row[data-id="${state.selected.id}"]`);
  if (row) row.focus();
}

// ------------------------------------------------------------------ about
function renderAbout() {
  const m = state.meta;
  const vp = m.validation.protein;
  const vg = m.validation.graph;
  $("fact-genes").textContent = int(m.n_genes);
  $("fact-top10").textContent = Math.round(vp.top10 * 100) + "%";
  $("validation-text").innerHTML =
    `SFARI added ${int(m.added_total)} genes between January 2024 and July 2026. ` +
    `${int(vp.n_new)} of them had a score from both models and no earlier label. ` +
    `<strong>${Math.round(vp.top10 * 100)}%</strong> of these were in the protein model's top 10% ` +
    `(${Math.round(vg.top10 * 100)}% for the network model), against 10% expected by chance.`;

  const counts = { candidates: 0, new: 0, known: 0, all: state.genes.length };
  for (const g of state.genes) {
    if (VIEWS.candidates.test(g)) counts.candidates++;
    if (VIEWS.new.test(g)) counts.new++;
    if (VIEWS.known.test(g)) counts.known++;
  }
  document.querySelectorAll("[data-count]").forEach((el) => { el.textContent = int(counts[el.dataset.count]); });

  $("sources").innerHTML = Object.values(m.sources)
    .map((s) => `<li><a href="${s.url}" target="_blank" rel="noopener">${esc(s.label)}</a>${s.version ? ` <span class="mono">${esc(s.version)}</span>` : ""}</li>`)
    .join("");
  renderChart();
}

function renderChart() {
  const key = state.chart === "protein" ? "pr" : "gr";
  const v = state.meta.validation[state.chart];
  const n = state["pool_" + key];
  const bins = new Array(10).fill(0);
  for (const g of state.genes) {
    if (g.role === "new" && g["c" + key]) bins[Math.min(9, Math.floor(((g["c" + key] - 1) / n) * 10))]++;
  }
  const total = bins.reduce((s, b) => s + b, 0);
  const expected = total / 10;
  const W = 560, H = 250, L = 34, R = 10, T = 18, B = 42;
  const yMax = Math.max(...bins, expected) * 1.12;
  const bw = (W - L - R) / 10;
  const y = (c) => T + (H - T - B) * (1 - c / yMax);
  const grid = [0, 20, 40, 60].filter((t) => t <= yMax)
    .map((t) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${t}</text>`).join("");
  const bars = bins.map((c, i) => {
    const x = L + i * bw + 3;
    return `<rect class="bar-new" x="${x}" y="${y(c)}" width="${bw - 6}" height="${y(0) - y(c)}" rx="2"/>
      <text class="lbl" x="${x + (bw - 6) / 2}" y="${y(c) - 5}" text-anchor="middle">${c}</text>
      <text x="${x + (bw - 6) / 2}" y="${H - B + 16}" text-anchor="middle">${i === 0 ? "top 10%" : `${i * 10}–${(i + 1) * 10}`}</text>`;
  }).join("");
  $("chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${total} genes added to SFARI after training, counted by decile of the ${state.chart === "protein" ? "protein" : "network"} ranking: ${bins.join(", ")} from the top decile down.">
    ${grid}${bars}
    <line class="expected" x1="${L}" x2="${W - R}" y1="${y(expected)}" y2="${y(expected)}"/>
    <text x="${W - R}" y="${y(expected) - 6}" text-anchor="end">expected by chance (${expected.toFixed(1)})</text>
    <text x="${(W + L) / 2}" y="${H - 6}" text-anchor="middle">position among the ${int(n)} genes unlabelled in 2024 (%)</text>
  </svg>`;
  $("chart-caption").textContent =
    `${total} genes added to SFARI after training, by where the ${state.chart === "protein" ? "protein" : "network"} model ranked them. ` +
    `${Math.round(v.top1 * 100)}% were in the top 1% and ${Math.round(v.top10 * 100)}% in the top 10%.`;
}

// ------------------------------------------------------------------ wiring
function wire() {
  let timer;
  $("q").addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => { state.query = e.target.value; refresh(); }, 90);
  });
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && state.rows[0]) select(state.rows[0], { open: true });
  });
  document.querySelectorAll(".filters [data-view]").forEach((b) => b.addEventListener("click", () => {
    state.view = b.dataset.view;
    document.querySelectorAll(".filters [data-view]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    refresh();
  }));
  $("sort").addEventListener("change", (e) => { state.sort = e.target.value; refresh(); });
  $("more").addEventListener("click", () => { state.shown += PAGE; renderList(); });
  $("search-all").addEventListener("click", () => document.querySelector('[data-view="all"]').click());
  $("list").addEventListener("click", (e) => {
    const row = e.target.closest(".row");
    if (row) select(state.byId.get(row.dataset.id), { open: true });
  });
  document.querySelectorAll(".chart-tabs [data-chart]").forEach((b) => b.addEventListener("click", () => {
    state.chart = b.dataset.chart;
    document.querySelectorAll(".chart-tabs [data-chart]").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    renderChart();
  }));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("detail").classList.contains("open")) closeDetail(); });
  addEventListener("hashchange", fromHash);
}

function fromHash() {
  const sym = decodeURIComponent(location.hash.slice(1)).toUpperCase();
  const g = sym && (state.bySymbol.get(sym) || state.byId.get(sym));
  if (g && g !== state.selected) select(g, { scroll: true });
  return Boolean(g);
}

async function main() {
  try {
    await load();
  } catch (err) {
    $("detail").innerHTML = `<p class="loading">Could not load the rankings (${esc(err.message)}). Reload the page to try again.</p>`;
    return;
  }
  wire();
  renderAbout();
  refresh();
  if (!fromHash()) {
    // Open on the best-ranked gene that SFARI added after training: a prediction that held up.
    select(state.rows.find((g) => g.role === "new") || state.rows[0]);
  }
}

main();
