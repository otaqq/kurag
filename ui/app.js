const $ = (id) => document.getElementById(id);
const dark = matchMedia("(prefers-color-scheme: dark)").matches;
let cy, data, center = "", shown = null;  // center: the section drawn around; shown: the section in the side panel
let namespaces = {}, root, current = "", path = "";  // current: the namespace drawn; path: where the level drop-downs are

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (e) { return null; }
}
const expanded = new Set(JSON.parse(store("kurag-open") || "[]"));

// One colour per file, stable across reloads
function color(file) {
  let h = 0;
  for (const c of file) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h}, 65%, ${dark ? 62 : 45}%)`;
}

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

// The sections around one section: a whole namespace can have thousands, too many to lay out in a browser or to read
function draw() {
  const elements = [
    ...data.nodes.map((n) => ({ data: { ...n, color: color(n.file), size: 14 + 6 * Math.sqrt(n.chunks), tip: place(n) },
      classes: n.id === center ? "center" : "" })),
    ...data.edges.map((e) => ({ data: { id: `${e.source}|${e.target}`, ...e } })),
  ];
  if (cy) cy.destroy();
  cy = cytoscape({
    container: $("cy"),
    elements,
    style: [
      { selector: "node", style: { "background-color": "data(color)", width: "data(size)", height: "data(size)",
        label: "data(label)", "font-size": 9, color: dark ? "#ddd" : "#333", "text-valign": "bottom", "text-margin-y": 3,
        "text-max-width": 120, "text-wrap": "ellipsis" } },
      { selector: "edge", style: { "curve-style": "haystack", width: "mapData(weight, 0.4, 0.95, 0.3, 3)", "line-color": dark ? "#556" : "#bbb",
        opacity: "mapData(weight, 0.4, 0.95, 0.25, 0.9)" } },
      { selector: ".faded", style: { opacity: 0.1 } },
      { selector: "node.center", style: { "border-width": 4, "border-color": dark ? "#fff" : "#000", "font-weight": "bold" } },
      { selector: "node:selected", style: { "border-width": 3, "border-color": dark ? "#fff" : "#000" } },
    ],
    layout: { name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 80, randomize: true },
    wheelSensitivity: 0.3,
  });
  cy.on("tap", "node", (ev) => open(ev.target.data()));
  cy.on("dbltap", "node", (ev) => around(ev.target.id()));
  cy.on("mouseover", "node", (ev) => {
    const n = ev.target;
    $("cy").title = n.data("tip");
    cy.elements().addClass("faded");
    n.closedNeighborhood().removeClass("faded");
  });
  cy.on("mouseout", "node", () => {
    cy.elements().removeClass("faded");
    $("cy").title = "";
  });
  const c = data.nodes.find((n) => n.id === center);
  $("stats").textContent = `${data.nodes.length} sections · ${data.edges.length} edges around “${c ? c.label : center}”`;
}

// The file and the heading or page of a section (its label shows keywords)
const place = (n) => (n.name ? `${n.file} · ${n.name}` : n.file);

function hint(text) {
  $("hint").textContent = text;
  $("hint").hidden = !text;
}

async function around(key) {
  center = key;
  $("minv").textContent = parseFloat($("min").value).toFixed(2);
  hint("…");
  const q = new URLSearchParams({ key, depth: $("depth").value, min_weight: $("min").value });
  data = await getJSON(`/ui/around/${encodeURIComponent(current)}?${q}`);
  hint(data.nodes.length > 1 ? "" : "No section is that similar to this one. Lower the edge similarity.");
  draw();
  for (const li of $("hits-list").children) li.classList.toggle("current", li.dataset.id === key);
  if ($("side").classList.contains("open")) open(data.nodes.find((n) => n.id === key));
}

// The list on the left: search hits, or the most linked sections of the namespace before any search
function showHits(title, items, note) {
  $("hits-title").textContent = title;
  $("hits").hidden = false;
  $("hits-list").replaceChildren(...items.map((h) => {
    const li = document.createElement("li");
    li.dataset.id = h.id;
    li.title = `${place(h)}\n${note(h)}`;
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = h.label;
    const file = document.createElement("span");
    file.className = "file";
    file.textContent = h.file.split("/").pop();
    li.append(label, file);
    li.onclick = () => {
      if (matchMedia("(max-width: 700px)").matches) $("hits").hidden = true;
      around(h.id);
    };
    return li;
  }));
}

async function search() {
  const q = $("q").value.trim();
  if (!q || !current) return;
  showHits("Searching…", [], () => "");
  const hits = await getJSON(`/ui/search/${encodeURIComponent(current)}?${new URLSearchParams({ q })}`);
  showHits(`${hits.length} sections for “${q}”`, hits, (h) => `score ${h.score}`);
  if (hits.length) around(hits[0].id);
}

async function hubs() {
  const namespace = current;
  const top = await getJSON(`/ui/hubs/${encodeURIComponent(namespace)}?min_weight=${$("min").value}`);
  if (namespace !== current || center) return;  // another namespace, or a search came first
  showHits("Most linked sections", top, (h) => `${h.links} links`);
}

async function open(node) {
  shown = node.id;
  $("side").classList.add("open");
  $("around").hidden = node.id === center;
  $("title").textContent = node.label;
  $("src").textContent = place(node);
  $("text").textContent = "…";
  const r = await getJSON(`/ui/section/${encodeURIComponent(current)}?key=${encodeURIComponent(node.id)}`);
  $("src").textContent = r.source ? `${r.source} · ${node.name || ""}` : place(node);
  $("text").textContent = r.text;
}

// A new namespace: nothing drawn until a search
function load() {
  store("kurag-namespace", current);
  if (cy) cy.destroy();
  cy = null;
  center = "";
  $("hits").hidden = true;
  $("side").classList.remove("open");
  $("stats").textContent = `${namespaces[current]} chunks`;
  hint(`Pick a section on the left, or search ${current}, to see the sections around it.`);
  hubs();
}

// ---- Namespace names as a tree: academic.cs.ml -> academic > cs > ml. Every level above a namespace is shown, as a level
// that cannot be drawn (language.fr.ja.book needs language, fr and ja above it, or its "book" could be any book).
function buildTree(names) {
  const top = { name: "", full: "", children: new Map() };
  for (const a of names) {
    let n = top;
    a.split(".").forEach((p, i, ps) => {
      if (!n.children.has(p)) n.children.set(p, { name: p, full: ps.slice(0, i + 1).join("."), children: new Map() });
      n = n.children.get(p);
    });
  }
  return top;
}
const kids = (n) => [...n.children.values()].sort((a, b) => a.name.localeCompare(b.name));
const total = (n) => (namespaces[n.full] || 0) + kids(n).reduce((s, c) => s + total(c), 0);
function nodeOf(full) {
  let n = root;
  for (const p of full ? full.split(".") : []) n = n.children.get(p);
  return n;
}
// A level that is not a namespace and has only one level below is joined with it ("ja.book"), so deep names stay short
function compact(n) {
  let label = n.name;
  while (!(n.full in namespaces) && n.children.size === 1) {
    n = kids(n)[0];
    label += "." + n.name;
  }
  return [label, n];
}

function choose(namespace) {
  current = path = namespace;
  if (matchMedia("(max-width: 700px)").matches) setTree(false);
  for (let i = namespace.indexOf("."); i > 0; i = namespace.indexOf(".", i + 1)) expanded.add(namespace.slice(0, i));
  render();
  load();
}

// ---- Tree on the left
function toggle(full) {
  expanded.has(full) ? expanded.delete(full) : expanded.add(full);
  store("kurag-open", JSON.stringify([...expanded]));
  renderTree();
}

function row(label, n, depthOpen) {
  const div = document.createElement("div");
  div.className = "row" + (n.full in namespaces ? " namespace" : "") + (n.full === current ? " current" : "");
  div.title = n.full;
  const tw = document.createElement("button");
  tw.className = "twisty";
  tw.textContent = n.children.size ? (depthOpen ? "▾" : "▸") : "";
  tw.disabled = !n.children.size;
  tw.onclick = (ev) => { ev.stopPropagation(); toggle(n.full); };
  const name = document.createElement("span");
  name.className = "name";
  name.textContent = label;
  const count = document.createElement("span");
  count.className = "count";
  count.textContent = total(n);
  div.append(tw, name, count);
  div.onclick = () => (n.full in namespaces ? choose(n.full) : toggle(n.full));
  return div;
}

function renderTree() {
  const q = $("tree-q").value.trim().toLowerCase();
  if (q) {  // filtering: the matching namespaces by their full names, no levels
    const ul = document.createElement("ul");
    for (const a of Object.keys(namespaces).sort().filter((a) => a.toLowerCase().includes(q))) {
      const li = document.createElement("li");
      li.append(row(a, nodeOf(a), false));
      ul.append(li);
    }
    $("tree-list").replaceChildren(ul);
    return;
  }
  const list = (n) => {
    const ul = document.createElement("ul");
    for (const c of kids(n)) {
      const [label, end] = compact(c);
      const open = expanded.has(end.full);
      const li = document.createElement("li");
      li.append(row(label, end, open));
      if (end.children.size && open) li.append(list(end));
      ul.append(li);
    }
    return ul;
  };
  $("tree-list").replaceChildren(list(root));
}

function setTree(open) {
  $("tree").classList.toggle("closed", !open);
  $("treebtn").setAttribute("aria-pressed", open);
  store("kurag-tree", open ? "open" : "closed");
}

// ---- One drop-down per level in the header: academic > cs > ml > ja.book
function descend(full) {
  let n = nodeOf(full);
  while (n.full && !(n.full in namespaces) && n.children.size === 1) n = kids(n)[0];
  if (n.full in namespaces) return choose(n.full);
  path = n.full;
  renderCrumbs();
}

function renderCrumbs() {
  const box = $("crumbs");
  box.replaceChildren();
  let n = root;
  while (n.children.size) {
    const here = n;
    const sel = document.createElement("select");
    // "…" at a level that is not a namespace: choose below; "—" at a namespace that has namespaces below: stay here
    sel.add(new Option(here.full in namespaces ? "—" : "…", ""));
    let next = null;
    for (const c of kids(here)) {
      const [label, end] = compact(c);
      sel.add(new Option(`${label} (${total(end)})`, end.full));
      if (path === end.full || path.startsWith(end.full + ".")) next = end;
    }
    sel.value = next ? next.full : "";
    sel.onchange = () => (sel.value ? descend(sel.value) : here.full ? choose(here.full) : null);
    if (box.children.length) {
      const sep = document.createElement("span");
      sep.className = "sep";
      sep.textContent = "›";
      box.append(sep);
    }
    box.append(sel);
    if (!next) break;
    n = next;
  }
}

function setMode(mode) {
  document.body.className = `mode-${mode}`;
  $("m-tree").setAttribute("aria-pressed", mode === "tree");
  $("m-crumbs").setAttribute("aria-pressed", mode === "crumbs");
  store("kurag-mode", mode);
}

function render() {
  renderTree();
  renderCrumbs();
}

(async () => {
  namespaces = await getJSON("/ui/namespaces");
  root = buildTree(Object.keys(namespaces));
  setMode(store("kurag-mode") === "crumbs" ? "crumbs" : "tree");
  setTree(store("kurag-tree") !== "closed" && !matchMedia("(max-width: 700px)").matches);
  $("m-tree").onclick = () => setMode("tree");
  $("m-crumbs").onclick = () => setMode("crumbs");
  $("treebtn").onclick = () => setTree($("tree").classList.contains("closed"));
  $("tree-q").oninput = renderTree;
  $("q").onkeydown = (ev) => { if (ev.key === "Enter") search(); };
  $("min").oninput = () => ($("minv").textContent = parseFloat($("min").value).toFixed(2));
  $("min").onchange = $("depth").onchange = () => center && around(center);
  $("hits-close").onclick = () => ($("hits").hidden = true);
  $("close").onclick = () => $("side").classList.remove("open");
  $("around").onclick = () => shown && around(shown);
  const saved = store("kurag-namespace");
  const first = saved && saved in namespaces ? saved : Object.keys(namespaces).sort()[0];
  if (first) choose(first);
  else render();
})();
