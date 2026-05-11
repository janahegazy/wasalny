const state = {
  network: null,
  selectedEdges: new Set(),
  forcedEdges: new Set(),
  selectedNodes: new Set(),
};

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 1) => Number.isFinite(n) ? n.toFixed(d) : "--";

async function getJSON(url) {
  const res = await fetch(url);
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || "Request failed");
  return data;
}

function project(node) {
  const nodes = state.network.nodes;
  const minLon = Math.min(...nodes.map(n => n.longitude));
  const maxLon = Math.max(...nodes.map(n => n.longitude));
  const minLat = Math.min(...nodes.map(n => n.latitude));
  const maxLat = Math.max(...nodes.map(n => n.latitude));
  const x = 70 + ((node.longitude - minLon) / (maxLon - minLon)) * 860;
  const y = 560 - ((node.latitude - minLat) / (maxLat - minLat)) * 480;
  return { x, y };
}

function shortName(name) {
  return name
    .replace("Cairo ", "")
    .replace("International ", "Intl. ")
    .replace("New Administrative Capital", "New Capital")
    .replace(" Military", "");
}

function populateSelects() {
  const nodes = [...state.network.nodes].sort((a, b) => a.name.localeCompare(b.name));
  for (const id of ["startNode", "endNode", "originNode"]) {
    $(id).innerHTML = nodes.map(n => `<option value="${n.id}">${n.id} · ${n.name}</option>`).join("");
  }
  $("startNode").value = "4";
  $("endNode").value = "3";
  $("originNode").value = "7";
}

function drawMap() {
  const svg = $("networkMap");
  const byId = Object.fromEntries(state.network.nodes.map(n => [n.id, n]));
  svg.innerHTML = "";

  for (const edge of state.network.edges) {
    const a = project(byId[edge.from]);
    const b = project(byId[edge.to]);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    line.classList.add("edge", edge.level || "clear");
    if (edge.potential) line.classList.add("potential");
    if (state.selectedEdges.has(edge.key)) line.classList.add("selected");
    if (state.forcedEdges.has(edge.key)) line.classList.add("forced");
    line.appendChild(title(`${edge.key}: ${fmt(edge.time)} min, ${Math.round(edge.ratio * 100)}% capacity`));
    svg.appendChild(line);
  }

  for (const node of state.network.nodes) {
    const p = project(node);
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.classList.add("node");
    if (node.critical) group.classList.add("critical");
    if (state.selectedNodes.has(node.id)) group.classList.add("selected");
    group.setAttribute("transform", `translate(${p.x},${p.y})`);
    group.innerHTML = `<circle r="16"></circle><text text-anchor="middle" y="4">${node.id}</text><text class="name" text-anchor="middle" y="33">${shortName(node.name)}</text>`;
    group.appendChild(title(`${node.name} · ${node.type}`));
    svg.appendChild(group);
  }
}

function title(text) {
  const t = document.createElementNS("http://www.w3.org/2000/svg", "title");
  t.textContent = text;
  return t;
}

function renderHotspots() {
  $("hotspots").innerHTML = state.network.hotspots.map(h => `
    <div class="item">
      <b>${h.edge}</b>
      <small>${Math.round(h.ratio * 100)}% capacity · ${h.level} · ${fmt(h.weight)} min</small>
    </div>
  `).join("");
}

function stats(items) {
  $("resultStats").innerHTML = items.map(([label, value]) => `
    <div class="stat"><strong>${value}</strong><span>${label}</span></div>
  `).join("");
}

function chips(items) {
  $("pathList").innerHTML = items.map(item => `<span class="chip">${item}</span>`).join("");
}

function details(items) {
  $("details").innerHTML = items.map(([label, value]) => `
    <div class="item"><b>${label}</b><small>${value}</small></div>
  `).join("");
}

async function loadNetwork(keepSelection = false) {
  const period = $("period").value;
  state.network = await getJSON(`/api/network?period=${period}`);
  if (!keepSelection) {
    state.selectedEdges.clear();
    state.forcedEdges.clear();
    state.selectedNodes.clear();
  }
  $("nodeCount").textContent = state.network.summary.nodes;
  $("roadCount").textContent = state.network.summary.existingEdges;
  $("criticalCount").textContent = state.network.summary.criticalNodes;
  $("signalCount").textContent = state.network.summary.signals;
  if (!$("startNode").options.length) populateSelects();
  renderHotspots();
  drawMap();
}

async function runRoute() {
  const params = new URLSearchParams({
    start: $("startNode").value,
    end: $("endNode").value,
    period: $("period").value,
    algorithm: $("algorithm").value,
    blocked: $("blocked").value,
    hour: $("hour").value,
  });
  const result = await getJSON(`/api/route?${params}`);
  state.selectedEdges = new Set(result.edges);
  state.forcedEdges.clear();
  state.selectedNodes = new Set(result.path);
  drawMap();

  $("resultTitle").textContent = result.algorithm;
  $("resultSummary").textContent = result.valid
    ? `${result.pathNames[0]} to ${result.pathNames[result.pathNames.length - 1]}`
    : "No route was found for those inputs.";
  stats([
    ["Travel Time", `${fmt(result.totalTime)} min`],
    ["Nodes Checked", result.nodesVisited],
    ["Roads Checked", result.edgesConsidered],
    ["Stops", result.path.length],
  ]);
  chips(result.path);
  details(result.pathNames.map((name, i) => [`Stop ${i + 1}`, name]));
}

async function runEmergency() {
  const params = new URLSearchParams({ origin: $("originNode").value, period: $("period").value });
  const result = await getJSON(`/api/emergency?${params}`);
  state.selectedEdges = new Set(result.route.edges);
  state.forcedEdges.clear();
  state.selectedNodes = new Set(result.route.path);
  drawMap();

  $("resultTitle").textContent = "Emergency Response";
  $("resultSummary").textContent = `Nearest hospital: ${result.hospitalName}`;
  stats([
    ["With Signals", `${fmt(result.withoutPreemption)} min`],
    ["Preempted", `${fmt(result.withPreemption)} min`],
    ["Saved", `${fmt(result.saved)} min`],
    ["Improvement", `${fmt(result.improvement)}%`],
  ]);
  chips(result.route.path);
  details(result.signals.length
    ? result.signals.map(s => [s.name, `${fmt(s.saved / 60)} min saved on ${s.edge}`])
    : [["Signals", "No signalized intersections on this path"]]);
}

async function runMst() {
  const params = new URLSearchParams({
    mode: $("mstMode").value,
    potential: $("mstMode").value === "cost" ? "true" : "false",
  });
  const result = await getJSON(`/api/mst?${params}`);
  state.selectedEdges = new Set(result.edges);
  state.forcedEdges = new Set(result.forced);
  state.selectedNodes.clear();
  drawMap();

  $("resultTitle").textContent = "Kruskal MST Plan";
  $("resultSummary").textContent = result.mode === "cost"
    ? "Expansion planning includes proposed roads."
    : "Existing-road backbone minimizing distance.";
  stats([
    ["MST Roads", result.edges.length],
    ["Distance", `${fmt(result.totalDistance)} km`],
    ["New Cost", `${fmt(result.totalCost, 0)} M EGP`],
    ["Maintenance", `${fmt(result.maintenance)} M EGP`],
  ]);
  chips(result.edges);
  details(result.newRoads.length
    ? result.newRoads.map(r => [r.key, `${fmt(r.distance)} km · ${fmt(r.cost, 0)} M EGP`])
    : [["New roads", "None selected in this mode"]]);
}

async function runTransit() {
  const result = await getJSON(`/api/transit?period=${$("period").value}`);
  state.selectedEdges.clear();
  state.forcedEdges.clear();
  state.selectedNodes.clear();
  drawMap();

  $("resultTitle").textContent = "Transit Allocation";
  $("resultSummary").textContent = `${result.fleet.buses} buses and ${result.fleet.trains} trains available in this period.`;
  stats([
    ["Network Coverage", `${fmt(result.coverage)}%`],
    ["Hourly Capacity", Math.round(result.totalHourlyCapacity).toLocaleString()],
    ["Hourly Demand", Math.round(result.estimatedHourlyDemand).toLocaleString()],
    ["Routes", result.routes.length],
  ]);
  chips(result.routes.slice(0, 8).map(r => r.id));
  details(result.routes.slice(0, 10).map(r => [
    `${r.id} · ${r.type}`,
    `${r.vehiclesPerHour}/hr · ${fmt(r.coverage, 0)}% coverage`,
  ]));
}

function wire() {
  $("period").addEventListener("change", () => loadNetwork(true));
  $("runRoute").addEventListener("click", runRoute);
  $("runEmergency").addEventListener("click", runEmergency);
  $("runMst").addEventListener("click", runMst);
  $("runTransit").addEventListener("click", runTransit);

  $("algorithm").addEventListener("change", () => {
    $("blockedWrap").classList.toggle("show", $("algorithm").value === "alternate");
    $("hourWrap").classList.toggle("show", $("algorithm").value === "time");
  });

  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
      tab.classList.add("active");
      $(`${tab.dataset.tab}Panel`).classList.add("active");
    });
  });
}

wire();
loadNetwork().then(() => {
  stats([
    ["Live Algorithms", "5"],
    ["Data Source", "Cairo"],
    ["Transit Routes", state.network.summary.transitRoutes],
    ["Road Signals", state.network.summary.signals],
  ]);
  renderHotspots();
}).catch(err => {
  $("resultTitle").textContent = "Startup Error";
  $("resultSummary").textContent = err.message;
});
