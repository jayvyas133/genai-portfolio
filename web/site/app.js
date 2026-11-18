// One page for the three portfolio APIs, behind the nginx gateway:
//   /api/documind/*  DocuMind      POST /ask
//   /api/research/*  Deep Research POST /research, GET /research/{id}[/events]
//   /api/sql/*       Text-to-SQL   POST /conversations, POST /conversations/{id}/messages
// Model text is always inserted with textContent, never innerHTML, so a
// reply can't inject markup into the page.

"use strict";

// ---------- helpers ----------

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else node[key] = value;
  }
  for (const child of children) node.append(child);
  return node;
}

// POST JSON and return the parsed body; throws Error with the API's detail.
async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return readJson(response);
}

async function readJson(response) {
  let data = null;
  try { data = await response.json(); } catch { /* not JSON, e.g. a 502 from nginx */ }
  if (!response.ok) {
    const detail = data && (data.detail || data.error);
    throw new Error(`${response.status}: ${typeof detail === "string" ? detail : response.statusText}`);
  }
  return data;
}

function busy(form, on) {
  for (const control of form.querySelectorAll("input, button")) control.disabled = on;
}

function seconds(ms) { return `${(ms / 1000).toFixed(1)} s`; }

// ---------- tabs ----------

const tabs = document.querySelectorAll("nav [role=tab]");
function showTab(name) {
  for (const tab of tabs) {
    const selected = tab.dataset.tab === name;
    tab.setAttribute("aria-selected", String(selected));
    document.getElementById(tab.dataset.tab).hidden = !selected;
  }
  history.replaceState(null, "", `#${name}`);
}
for (const tab of tabs) tab.addEventListener("click", () => showTab(tab.dataset.tab));
if (location.hash && document.getElementById(location.hash.slice(1))) showTab(location.hash.slice(1));

// ---------- DocuMind ----------

const docForm = document.getElementById("documind-form");
const docOut = document.getElementById("documind-out");

docForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = docForm.question.value.trim();
  docOut.replaceChildren(el("p", { class: "working", textContent: "Searching the documents..." }));
  busy(docForm, true);
  try {
    const data = await postJson("/api/documind/ask", { question });
    const parts = [el("p", { textContent: data.refused ? data.message : data.answer })];
    if (data.citations.length) {
      parts.push(el("ol", { class: "sources" }, ...data.citations.map((c) =>
        el("li", { value: c.number, textContent: `${c.document_title} - ${c.section}` }))));
    }
    parts.push(el("p", { class: "meta", textContent:
      `${data.refused ? `Refused (${data.refusal_reason})` : "Answered"} in ${seconds(data.latency_ms)}` }));
    docOut.replaceChildren(...parts);
  } catch (err) {
    docOut.replaceChildren(el("p", { class: "error", textContent: err.message }));
  } finally {
    busy(docForm, false);
  }
});

// ---------- Deep Research ----------

const resForm = document.getElementById("research-form");
const resProgress = document.getElementById("research-progress");
const resOut = document.getElementById("research-out");

resForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = resForm.question.value.trim();
  resProgress.replaceChildren();
  resOut.replaceChildren();
  busy(resForm, true);
  try {
    const job = await postJson("/api/research/research", { question });
    resProgress.append(el("li", { textContent: "Queued" }));
    await followJob(job.id);
  } catch (err) {
    resOut.replaceChildren(el("p", { class: "error", textContent: err.message }));
    busy(resForm, false);
  }
});

// Stream the job's events (the API replays past ones first), then fetch the report.
function followJob(id) {
  return new Promise((resolve) => {
    const started = performance.now();
    const source = new EventSource(`/api/research/research/${id}/events`);
    const add = (text, final = false) =>
      resProgress.append(el("li", { class: final ? "final" : "", textContent: text }));

    source.addEventListener("started", () => add("Started"));
    source.addEventListener("progress", (e) => {
      const data = JSON.parse(e.data);
      add(data.node ? `[${data.node}] ${data.message}` : data.message);
    });
    for (const type of ["done", "failed", "cancelled"]) {
      source.addEventListener(type, async (e) => {
        source.close();  // final event: stop EventSource from reconnecting
        add(`${type} after ${seconds(performance.now() - started)}`, true);
        if (type === "done") await showReport(id);
        else resOut.replaceChildren(el("p", { class: "error", textContent: JSON.parse(e.data).message }));
        busy(resForm, false);
        resolve();
      });
    }
  });
}

async function showReport(id) {
  try {
    const job = await readJson(await fetch(`/api/research/research/${id}`));
    const report = job.report;
    const parts = report.text.split(/\n{2,}/).map((p) => el("p", { textContent: p }));
    if (report.sources.length) {
      parts.push(el("ol", { class: "sources" }, ...report.sources.map((s) =>
        el("li", {}, el("a", { href: s.url, target: "_blank", rel: "noopener",
          textContent: `${s.title} - ${s.section}` })))));
    }
    resOut.replaceChildren(...parts);
  } catch (err) {
    resOut.replaceChildren(el("p", { class: "error", textContent: err.message }));
  }
}

// ---------- Text-to-SQL chat ----------

const sqlForm = document.getElementById("sql-form");
const chat = document.getElementById("chat");
let conversationId = null;

document.getElementById("sql-new").addEventListener("click", () => {
  conversationId = null;
  chat.replaceChildren();
  sqlForm.question.focus();
});

function resultTable(columns, rows) {
  const head = el("tr", {}, ...columns.map((c) => el("th", { textContent: c })));
  const body = rows.slice(0, 50).map((row) =>
    el("tr", {}, ...row.map((v) => el("td", { textContent: v === null ? "NULL" : String(v) }))));
  return el("div", { class: "table-wrap" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
}

sqlForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = sqlForm.question.value.trim();
  sqlForm.question.value = "";
  chat.append(el("div", { class: "msg user", textContent: question }));
  const reply = el("div", { class: "msg bot" }, el("p", { class: "working", textContent: "Thinking..." }));
  chat.append(reply);
  // Disabled while a turn runs: the API allows one turn at a time per conversation (409).
  busy(sqlForm, true);
  try {
    if (conversationId === null) conversationId = (await postJson("/api/sql/conversations")).id;
    const turn = await postJson(`/api/sql/conversations/${conversationId}/messages`, { question });
    const parts = [el("p", { textContent: turn.answer || "(no answer)" })];
    const details = el("details", {}, el("summary", {
      textContent: turn.sql ? `SQL and ${turn.rows.length}${turn.truncated ? "+" : ""} row(s)` : "Details",
    }));
    if (turn.sql) details.append(el("pre", { textContent: turn.sql }));
    if (turn.columns.length) details.append(resultTable(turn.columns, turn.rows));
    if (turn.error) details.append(el("p", { class: "error", textContent: `${turn.error_kind}: ${turn.error}` }));
    parts.push(details, el("p", { class: "meta", textContent:
      `${seconds(turn.latency_ms)}${turn.attempts > 1 ? `, repaired after ${turn.attempts} attempts` : ""}` }));
    reply.replaceChildren(...parts);
  } catch (err) {
    reply.replaceChildren(el("p", { class: "error", textContent: err.message }));
  } finally {
    busy(sqlForm, false);
    sqlForm.question.focus();
  }
});
