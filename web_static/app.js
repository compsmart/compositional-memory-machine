const $ = (selector) => document.querySelector(selector);
const els = {
  statusStrip: $("#statusStrip"),
  metricsGrid: $("#metricsGrid"),
  factList: $("#factList"),
  factCountBadge: $("#factCountBadge"),
  chatForm: $("#chatForm"),
  chatInput: $("#chatInput"),
  chatHistory: $("#chatHistory"),
  routeBadge: $("#routeBadge"),
  ingestForm: $("#ingestForm"),
  ingestText: $("#ingestText"),
  ingestDomain: $("#ingestDomain"),
  ingestSource: $("#ingestSource"),
  ingestResult: $("#ingestResult"),
  memoryBankSelect: $("#memoryBankSelect"),
  loadBankButton: $("#loadBankButton"),
  refreshButton: $("#refreshButton"),
  resetDemoButton: $("#resetDemoButton"),
  spinButton: $("#spinButton"),
  memoryFilter: $("#memoryFilter"),
  domainFilter: $("#domainFilter"),
  sourceFilter: $("#sourceFilter"),
  memoryScene: $("#memoryScene"),
  memoryFallback: $("#memoryFallback"),
  nodeInspector: $("#nodeInspector"),
  compositionalCard: $("#compositionalCard"),
};

let latestFacts = [];
let filteredFacts = [];
let memoryGraph = null;
let chatHistory = [];
let memoryBanks = [];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function setBusy(button, busy) {
  if (!button) return;
  button.disabled = busy;
}

function renderStatus(status) {
  const keyState = status.google_api_key ? "Gemini key: yes" : "Gemini key: no";
  els.statusStrip.innerHTML = [
    ["Facts", status.stored_facts],
    ["Graph", status.graph_facts],
    ["Records", status.memory_records],
    ["Chunk budget", status.chunk_budget],
    ["Dimension", status.dim],
    ["Bank", status.current_memory_bank || "Seed demo"],
    ["Demo", status.demo_value],
    ["Key", keyState],
  ]
    .map(([label, value]) => `<span class="status-pill">${label}: ${escapeHtml(String(value))}</span>`)
    .join("");

  els.metricsGrid.innerHTML = [
    ["Facts", status.stored_facts],
    ["Graph", status.graph_facts],
    ["Records", status.memory_records],
    ["Chunk budget", status.chunk_budget],
    ["D", status.dim],
    ["Bank", status.current_memory_bank || "Seed demo"],
  ]
    .map(([label, value]) => `<div class="metric"><strong>${escapeHtml(String(value))}</strong><span>${escapeHtml(label)}</span></div>`)
    .join("");
}

function renderMemoryBanks(payload) {
  memoryBanks = Array.isArray(payload?.banks) ? payload.banks : [];
  const current = payload?.current_bank_id || "seed";
  const previous = els.memoryBankSelect.value;
  els.memoryBankSelect.innerHTML = memoryBanks
    .map((bank) => {
      const countSuffix = Number.isFinite(bank.fact_count) ? ` (${bank.fact_count} facts)` : "";
      return `<option value="${escapeHtml(bank.id)}">${escapeHtml(bank.label || `${bank.id}${countSuffix}`)}</option>`;
    })
    .join("");
  const preferred = memoryBanks.some((bank) => bank.id === previous) ? previous : current;
  if (preferred) {
    els.memoryBankSelect.value = preferred;
  }
}

function renderFacts(facts) {
  latestFacts = facts;
  populateFilterOptions(facts);
  applyMemoryFilters();
}

function applyMemoryFilters() {
  const query = els.memoryFilter.value.trim().toLowerCase();
  const domain = els.domainFilter.value;
  const source = els.sourceFilter.value;
  filteredFacts = latestFacts.filter((fact) => {
    if (domain && factDomain(fact) !== domain) return false;
    if (source && factSource(fact) !== source) return false;
    if (!query) return true;
    return [
      fact.subject,
      fact.relation,
      fact.object,
      fact.kind,
      factSource(fact),
      factDomain(fact),
    ].some((value) => String(value || "").toLowerCase().includes(query));
  });
  renderFilteredFacts(filteredFacts);
}

function renderFilteredFacts(facts) {
  els.factCountBadge.textContent = `${facts.length} facts`;
  if (!latestFacts.length) {
    els.factList.innerHTML = `<div class="result-block empty">No facts stored</div>`;
    if (memoryGraph) memoryGraph.setFacts([]);
    return;
  }
  if (!facts.length) {
    els.factList.innerHTML = `<div class="result-block empty">No facts match the current filters</div>`;
    if (memoryGraph) memoryGraph.setFacts([]);
    return;
  }
  els.factList.innerHTML = facts
    .slice()
    .reverse()
    .map(
      (fact) => `
        <article class="fact-card" data-node="${escapeHtml(fact.subject)}">
          <strong>${escapeHtml(fact.subject)} &rarr; ${escapeHtml(fact.object)}</strong>
          <p>${escapeHtml(fact.relation)} - confidence ${Number(fact.confidence).toFixed(2)}</p>
          <small>${escapeHtml(factDomain(fact))} / ${escapeHtml(factSource(fact))}</small>
        </article>`
    )
    .join("");
  if (memoryGraph) memoryGraph.setFacts(facts);
}

function populateFilterOptions(facts) {
  const currentDomain = els.domainFilter.value;
  const currentSource = els.sourceFilter.value;
  const domains = uniqueSorted(facts.map(factDomain));
  const sources = uniqueSorted(facts.map(factSource));
  els.domainFilter.innerHTML = `<option value="">All domains</option>${domains
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`)
    .join("")}`;
  els.sourceFilter.innerHTML = `<option value="">All sources</option>${sources
    .map((item) => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`)
    .join("")}`;
  if (domains.includes(currentDomain)) els.domainFilter.value = currentDomain;
  if (sources.includes(currentSource)) els.sourceFilter.value = currentSource;
}

function factDomain(fact) {
  return fact.domain || fact.metadata?.domain || "unknown";
}

function factSource(fact) {
  return fact.source || "unknown";
}

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b));
}

function renderChatHistory(history) {
  chatHistory = history || [];
  const lastAssistant = [...chatHistory].reverse().find((message) => message.role === "assistant");
  els.routeBadge.textContent = lastAssistant?.route || "ready";
  els.chatHistory.classList.toggle("empty", chatHistory.length === 0);
  if (!chatHistory.length) {
    els.chatHistory.innerHTML = "No messages yet";
    return;
  }
  els.chatHistory.innerHTML = chatHistory.map(renderChatMessage).join("");
  els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
  if (memoryGraph && lastAssistant?.evidence?.length) {
    memoryGraph.focusFromAnswer(String(lastAssistant.graph_target || lastAssistant.text || ""), lastAssistant.evidence);
  }
}

function renderChatMessage(message) {
  const meta = [];
  if (message.route) meta.push(`route ${escapeHtml(message.route)}`);
  if (Number.isFinite(message.confidence)) meta.push(`confidence ${Number(message.confidence).toFixed(3)}`);
  const evidence = Array.isArray(message.evidence) ? message.evidence.slice(0, 3) : [];
  const chainPath = Array.isArray(message.chain_path) ? message.chain_path : [];
  const alternatives = Array.isArray(message.alternatives) ? message.alternatives.slice(0, 3) : [];
  return `
    <article class="chat-message ${escapeHtml(message.role || "assistant")}">
      <div class="chat-message-head">
        <strong>${escapeHtml(message.role === "user" ? "You" : "Assistant")}</strong>
        ${meta.length ? `<span class="meta-line">${meta.join(" - ")}</span>` : ""}
      </div>
      <div class="chat-bubble">${escapeHtml(message.text || "")}</div>
      ${
        message.graph_target
          ? `<div class="meta-line">graph target: ${escapeHtml(message.graph_target)}</div>`
          : ""
      }
      ${
        chainPath.length
          ? `<div class="meta-line">chain: ${chainPath.map((item) => escapeHtml(String(item))).join(" -> ")}</div>`
          : ""
      }
      ${
        evidence.length
          ? `<ul class="query-evidence">${evidence
              .map(
                (fact) =>
                  `<li>${escapeHtml(fact.subject)} ${escapeHtml(fact.relation)} ${escapeHtml(fact.object)} <span class="meta-line">(score ${Number(fact.score).toFixed(3)}${fact.chunk_id ? `, chunk ${escapeHtml(String(fact.chunk_id))}` : ""})</span></li>`
              )
              .join("")}</ul>`
          : ""
      }
      ${
        alternatives.length
          ? `<div class="meta-line">alternatives: ${alternatives
              .map((item) => `${escapeHtml(item.token)} (${Number(item.probability).toFixed(2)})`)
              .join(", ")}</div>`
          : ""
      }
    </article>
  `;
}

function renderChatError(error) {
  renderChatHistory([
    ...chatHistory,
    {
      role: "assistant",
      text: error.message || String(error),
      route: "error",
    },
  ]);
}

function renderCompositional(payload) {
  els.compositionalCard.innerHTML = `
    <strong>Compositional Demo</strong>
    <p>${escapeHtml(payload.question)}</p>
    <div class="value-grid">
      <div class="value-metric">
        <span>Expected</span>
        <strong>${escapeHtml(payload.expected_value)}</strong>
      </div>
      <div class="value-metric">
        <span>Entity</span>
        <strong>${escapeHtml(payload.entity)}</strong>
      </div>
      <div class="value-metric">
        <span>HRR Native</span>
        <strong>${escapeHtml(payload.hrr_native.text)}</strong>
      </div>
      <div class="value-metric">
        <span>Linear Head</span>
        <strong>${escapeHtml(payload.linear.text)}</strong>
      </div>
    </div>
    <div class="result-block">${escapeHtml(payload.answer)}</div>
  `;
}

function renderIngestResult(payload) {
  els.ingestResult.classList.remove("empty");
  els.ingestResult.textContent = JSON.stringify(payload, null, 2);
}

function renderError(target, error) {
  target.classList.remove("empty");
  target.innerHTML = `<span class="error">${escapeHtml(error.message || String(error))}</span>`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

class MemoryGraph3D {
  constructor(container, inspector) {
    this.container = container;
    this.inspector = inspector;
    this.facts = [];
    this.nodes = new Map();
    this.nodeMeshes = [];
    this.edgeObjects = [];
    this.labelObjects = [];
    this.labels = new Map();
    this.selectedNode = "";
    this.target = null;
    this.distance = 22;
    this.initialDistance = 22;
    this.theta = 0.8;
    this.phi = 1.1;
    this.pan = null;
    this.autoSpin = true;
    this.pointer = { active: false, x: 0, y: 0, button: 0, moved: false };
  }

  async init() {
    this.THREE = await import("https://unpkg.com/three@0.164.1/build/three.module.js");
    const THREE = this.THREE;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xfbfaf6);
    this.group = new THREE.Group();
    this.scene.add(this.group);
    this.camera = new THREE.PerspectiveCamera(48, 1, 0.1, 1000);
    this.target = new THREE.Vector3(0, 0, 0);
    this.pan = new THREE.Vector3(0, 0, 0);
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const light = new THREE.DirectionalLight(0xffffff, 0.9);
    light.position.set(10, 16, 8);
    this.scene.add(light);
    this.bindEvents();
    this.resize();
    this.updateCamera();
    this.animate();
  }

  bindEvents() {
    this.container.addEventListener("contextmenu", (event) => event.preventDefault());
    this.container.addEventListener("pointerdown", (event) => {
      this.pointer = { active: true, x: event.clientX, y: event.clientY, button: event.button, moved: false };
      this.container.setPointerCapture(event.pointerId);
    });
    this.container.addEventListener("pointermove", (event) => {
      if (!this.pointer.active) return;
      const dx = event.clientX - this.pointer.x;
      const dy = event.clientY - this.pointer.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) this.pointer.moved = true;
      if (event.shiftKey || this.pointer.button === 2) {
        this.panView(dx, dy);
      } else {
        this.theta -= dx * 0.006;
        this.phi = clamp(this.phi - dy * 0.006, 0.18, Math.PI - 0.18);
      }
      this.pointer.x = event.clientX;
      this.pointer.y = event.clientY;
      this.updateCamera();
    });
    this.container.addEventListener("pointerup", (event) => {
      if (!this.pointer.moved) this.pickNode(event);
      this.pointer.active = false;
    });
    this.container.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        this.distance = clamp(this.distance * (1 + event.deltaY * 0.001), 5, 90);
        this.updateCamera();
      },
      { passive: false }
    );
    window.addEventListener("keydown", (event) => {
      if (isEditableTarget(event.target)) return;
      if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        this.moveThroughScene(event.key === "ArrowUp" ? 1 : -1);
      } else {
        this.moveSideways(event.key === "ArrowRight" ? 1 : -1);
      }
    });
    window.addEventListener("resize", () => this.resize());
  }

  setFacts(facts) {
    this.facts = facts.slice(-600);
    this.clearGraph();
    this.buildGraph();
    this.renderInspector(this.selectedNode);
  }

  clearGraph() {
    for (const item of [...this.nodeMeshes, ...this.edgeObjects, ...this.labelObjects]) {
      item.geometry?.dispose?.();
      item.material?.map?.dispose?.();
      if (Array.isArray(item.material)) {
        item.material.forEach((mat) => mat.dispose?.());
      } else {
        item.material?.dispose?.();
      }
      this.group.remove(item);
    }
    this.nodes.clear();
    this.nodeMeshes = [];
    this.edgeObjects = [];
    this.labelObjects = [];
    this.labels.clear();
  }

  buildGraph() {
    const THREE = this.THREE;
    if (!THREE) return;
    const names = [];
    const seen = new Set();
    for (const fact of this.facts) {
      for (const name of [fact.subject, fact.object]) {
        if (!seen.has(name)) {
          seen.add(name);
          names.push(name);
        }
      }
    }
    const radius = Math.max(4, Math.sqrt(names.length) * 2.2);
    names.forEach((name, index) => {
      const pos = this.nodePosition(name, index, names.length, radius);
      const degree = this.facts.filter((fact) => fact.subject === name || fact.object === name).length;
      const color = degree > 3 ? 0x285f9d : degree > 1 ? 0x2f6d52 : 0xa66b16;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(clamp(0.22 + degree * 0.025, 0.22, 0.5), 24, 16),
        new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.08 })
      );
      mesh.position.copy(pos);
      mesh.userData.nodeName = name;
      this.group.add(mesh);
      this.nodeMeshes.push(mesh);
      this.nodes.set(name, { name, position: pos, mesh, facts: [] });
      const label = this.makeLabel(name, pos);
      label.visible = this.shouldShowBaseLabel(name, degree, names.length);
      this.labels.set(name, { sprite: label, baseVisible: label.visible });
      this.labelObjects.push(label);
      if (label.visible) {
        this.group.add(label);
      }
    });

    for (const fact of this.facts) {
      const source = this.nodes.get(fact.subject);
      const target = this.nodes.get(fact.object);
      if (!source || !target) continue;
      source.facts.push(fact);
      target.facts.push(fact);
      const geometry = new THREE.BufferGeometry().setFromPoints([source.position, target.position]);
      const line = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: 0x285f9d, transparent: true, opacity: 0.28 })
      );
      line.userData.fact = fact;
      this.group.add(line);
      this.edgeObjects.push(line);
    }
    this.updateHighlights();
    this.updateLabelVisibility();
  }

  nodePosition(name, index, count, radius) {
    const THREE = this.THREE;
    const h = hashCode(name);
    const offset = 2 / Math.max(1, count);
    const y = index * offset - 1 + offset / 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = index * 2.399963 + (h % 997) * 0.001;
    const jitter = 0.75 + ((h >>> 8) % 100) / 300;
    return new THREE.Vector3(
      Math.cos(angle) * r * radius * jitter,
      y * radius * jitter,
      Math.sin(angle) * r * radius * jitter
    );
  }

  makeLabel(text, position) {
    const THREE = this.THREE;
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgba(255,253,248,0.86)";
    roundRect(ctx, 4, 10, 248, 40, 8);
    ctx.fill();
    ctx.fillStyle = "#18201d";
    ctx.font = "700 20px system-ui";
    ctx.fillText(text.length > 22 ? `${text.slice(0, 20)}...` : text, 14, 37);
    const texture = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    sprite.position.copy(position).add(new THREE.Vector3(0.4, 0.34, 0));
    sprite.scale.set(2.8, 0.7, 1);
    return sprite;
  }

  pickNode(event) {
    if (!this.nodeMeshes.length) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const hit = this.raycaster.intersectObjects(this.nodeMeshes, false)[0];
    if (hit) this.selectNode(hit.object.userData.nodeName, { focus: false });
  }

  selectNode(name, options = {}) {
    if (!this.nodes.has(name)) return;
    this.selectedNode = name;
    this.updateHighlights();
    this.renderInspector(name);
    if (options.focus !== false) this.focusNode(name);
  }

  focusFromAnswer(answer, evidence) {
    const answerText = String(answer || "").trim().toLowerCase();
    let target = "";
    if (answerText) {
      for (const name of this.nodes.keys()) {
        if (name.toLowerCase() === answerText || answerText.includes(name.toLowerCase())) {
          target = name;
          break;
        }
      }
    }
    if (!target && evidence && evidence.length) {
      target = evidence[0].object || evidence[0].subject;
    }
    if (target) this.selectNode(target);
  }

  focusNode(name) {
    const node = this.nodes.get(name);
    if (!node) return;
    this.target.copy(node.position);
    this.distance = clamp(this.distance, 8, 24);
    this.updateCamera();
  }

  shouldShowBaseLabel(name, degree, totalNodes) {
    return totalNodes <= 48 || degree > 1 || name === this.selectedNode;
  }

  updateLabelVisibility() {
    if (!this.camera || !this.THREE) return;
    const zoomedIn = this.distance <= this.initialDistance / 1.8;
    const frustum = new this.THREE.Frustum();
    const projection = new this.THREE.Matrix4().multiplyMatrices(
      this.camera.projectionMatrix,
      this.camera.matrixWorldInverse
    );
    frustum.setFromProjectionMatrix(projection);
    const worldPosition = new this.THREE.Vector3();
    for (const [name, label] of this.labels.entries()) {
      const node = this.nodes.get(name);
      if (node) node.mesh.getWorldPosition(worldPosition);
      const connectedToSelected =
        this.selectedNode &&
        node?.facts.some((fact) => fact.subject === this.selectedNode || fact.object === this.selectedNode);
      const visible = Boolean(
        label.baseVisible ||
          name === this.selectedNode ||
          connectedToSelected ||
          (zoomedIn && node && frustum.containsPoint(worldPosition))
      );
      if (visible && !label.sprite.parent) this.group.add(label.sprite);
      if (!visible && label.sprite.parent) this.group.remove(label.sprite);
      label.sprite.visible = visible;
    }
  }

  updateHighlights() {
    const selected = this.selectedNode;
    const related = new Set();
    if (selected) {
      for (const fact of this.facts) {
        if (fact.subject === selected) related.add(fact.object);
        if (fact.object === selected) related.add(fact.subject);
      }
    }
    for (const mesh of this.nodeMeshes) {
      const name = mesh.userData.nodeName;
      const isSelected = name === selected;
      const isRelated = related.has(name);
      mesh.scale.setScalar(isSelected ? 1.9 : isRelated ? 1.35 : selected ? 0.75 : 1);
      mesh.material.opacity = selected && !isSelected && !isRelated ? 0.55 : 1;
      mesh.material.transparent = true;
    }
    for (const line of this.edgeObjects) {
      const fact = line.userData.fact;
      const active = selected && (fact.subject === selected || fact.object === selected);
      line.material.opacity = active ? 0.9 : selected ? 0.12 : 0.28;
      line.material.color.set(active ? 0xa66b16 : 0x285f9d);
    }
    this.updateLabelVisibility();
  }

  renderInspector(name) {
    if (!name || !this.nodes.has(name)) {
      this.inspector.classList.add("empty");
      this.inspector.textContent = "Select a node";
      return;
    }
    const facts = this.nodes.get(name).facts;
    this.inspector.classList.remove("empty");
    this.inspector.innerHTML = `
      <strong>${escapeHtml(name)}</strong>
      <div class="meta-line">${facts.length} connection${facts.length === 1 ? "" : "s"}</div>
      <ul>
        ${facts
          .slice(0, 12)
          .map((fact) => `<li>${escapeHtml(fact.subject)} <span>${escapeHtml(fact.relation)}</span> ${escapeHtml(fact.object)}</li>`)
          .join("")}
      </ul>
    `;
  }

  panView(dx, dy) {
    const THREE = this.THREE;
    const direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);
    const right = new THREE.Vector3().crossVectors(direction, this.camera.up).normalize();
    const up = new THREE.Vector3().copy(this.camera.up).normalize();
    const scale = this.distance * 0.0018;
    this.target.add(right.multiplyScalar(-dx * scale));
    this.target.add(up.multiplyScalar(dy * scale));
  }

  moveThroughScene(directionSign) {
    const THREE = this.THREE;
    const direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);
    const step = clamp(this.distance * 0.12, 0.8, 5);
    this.target.add(direction.multiplyScalar(directionSign * step));
    this.updateCamera();
  }

  moveSideways(directionSign) {
    const THREE = this.THREE;
    const direction = new THREE.Vector3();
    this.camera.getWorldDirection(direction);
    const right = new THREE.Vector3().crossVectors(direction, this.camera.up).normalize();
    const step = clamp(this.distance * 0.12, 0.8, 5);
    this.target.add(right.multiplyScalar(directionSign * step));
    this.updateCamera();
  }

  updateCamera() {
    const THREE = this.THREE;
    if (!THREE) return;
    const x = this.distance * Math.sin(this.phi) * Math.cos(this.theta);
    const y = this.distance * Math.cos(this.phi);
    const z = this.distance * Math.sin(this.phi) * Math.sin(this.theta);
    this.camera.position.set(this.target.x + x, this.target.y + y, this.target.z + z);
    this.camera.lookAt(this.target);
    this.updateLabelVisibility();
  }

  resize() {
    if (!this.renderer) return;
    const rect = this.container.getBoundingClientRect();
    this.camera.aspect = rect.width / Math.max(1, rect.height);
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(rect.width, rect.height, false);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (this.group && this.autoSpin) this.group.rotation.y += 0.0015;
    this.renderer.render(this.scene, this.camera);
  }

  setAutoSpin(enabled) {
    this.autoSpin = enabled;
  }
}

function hashCode(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function isEditableTarget(target) {
  if (!target) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function roundRect(ctx, x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

async function initMemoryGraph() {
  memoryGraph = new MemoryGraph3D(els.memoryScene, els.nodeInspector);
  try {
    await memoryGraph.init();
    if (filteredFacts.length || latestFacts.length) memoryGraph.setFacts(filteredFacts.length ? filteredFacts : latestFacts);
  } catch (error) {
    els.memoryScene.innerHTML = `<div class="scene-fallback error">3D memory failed to load: ${escapeHtml(error.message || error)}</div>`;
    memoryGraph = null;
  }
}

async function refresh() {
  const [status, facts, compositional, chat, banks] = await Promise.all([
    api("/api/status"),
    api("/api/facts"),
    api("/api/demo/compositional"),
    api("/api/chat/history"),
    api("/api/memory-banks"),
  ]);
  renderStatus(status);
  renderFacts(facts.facts || []);
  renderCompositional(compositional);
  renderChatHistory(chat.history || []);
  renderMemoryBanks(banks);
}

els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = els.chatInput.value.trim();
  if (!message) return;
  const button = event.submitter;
  setBusy(button, true);
  try {
    const payload = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    renderChatHistory(payload.history || []);
    renderStatus(payload.status);
    renderFacts(payload.facts?.facts || []);
    els.chatInput.value = "";
  } catch (error) {
    renderChatError(error);
  } finally {
    setBusy(button, false);
  }
});

els.ingestForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  setBusy(button, true);
  try {
    const payload = await api("/api/ingest/text", {
      method: "POST",
      body: JSON.stringify({
        text: els.ingestText.value,
        domain: els.ingestDomain.value || "web",
        source: els.ingestSource.value || "web-ui",
      }),
    });
    renderIngestResult(payload.ingestion);
    await refresh();
  } catch (error) {
    renderError(els.ingestResult, error);
  } finally {
    setBusy(button, false);
  }
});

els.resetDemoButton.addEventListener("click", async () => {
  setBusy(els.resetDemoButton, true);
  try {
    const payload = await api("/api/demo/reset", { method: "POST", body: JSON.stringify({}) });
    renderIngestResult({ reset: true, status: payload.status });
    renderStatus(payload.status);
    renderFacts(payload.facts?.facts || []);
    renderCompositional(payload.compositional);
    renderChatHistory(payload.chat?.history || []);
    renderMemoryBanks(payload.memory_banks);
  } catch (error) {
    renderError(els.ingestResult, error);
  } finally {
    setBusy(els.resetDemoButton, false);
  }
});

els.loadBankButton.addEventListener("click", async () => {
  setBusy(els.loadBankButton, true);
  try {
    const payload = await api("/api/memory-bank/select", {
      method: "POST",
      body: JSON.stringify({ bank_id: els.memoryBankSelect.value }),
    });
    renderIngestResult({
      bank: payload.selected_bank_label,
      loaded_archive_facts: payload.loaded_archive_facts,
      status: payload.status,
    });
    renderStatus(payload.status);
    renderFacts(payload.facts?.facts || []);
    renderCompositional(payload.compositional);
    renderChatHistory(payload.chat?.history || []);
    renderMemoryBanks(payload.memory_banks);
  } catch (error) {
    renderError(els.ingestResult, error);
  } finally {
    setBusy(els.loadBankButton, false);
  }
});

els.refreshButton.addEventListener("click", () => refresh().catch((error) => renderChatError(error)));
els.memoryFilter.addEventListener("input", applyMemoryFilters);
els.domainFilter.addEventListener("change", applyMemoryFilters);
els.sourceFilter.addEventListener("change", applyMemoryFilters);

els.factList.addEventListener("click", (event) => {
  const card = event.target.closest(".fact-card");
  if (!card || !memoryGraph) return;
  memoryGraph.selectNode(card.dataset.node);
});

els.spinButton.addEventListener("click", () => {
  const enabled = !els.spinButton.classList.contains("active");
  els.spinButton.classList.toggle("active", enabled);
  els.spinButton.title = enabled ? "Pause auto-spin" : "Resume auto-spin";
  els.spinButton.innerHTML = enabled
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  if (memoryGraph) memoryGraph.setAutoSpin(enabled);
});

els.ingestDomain.value = "history";
els.ingestSource.value = "web-ui";

initMemoryGraph();
refresh().catch((error) => renderChatError(error));
