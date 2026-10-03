// Evaluate CSV tab: uploads a CSV to /api/evaluate and renders the streamed
// NDJSON results as they arrive.

const fileInput = document.getElementById("csv-file");
const dropzone = document.getElementById("dropzone");
const fileNameEl = document.getElementById("file-name");
const csvText = document.getElementById("csv-text");
const runBtn = document.getElementById("run-eval");
const cancelBtn = document.getElementById("cancel-eval");
const evalError = document.getElementById("eval-error");
const progressEl = document.getElementById("progress");
const progressFill = document.getElementById("progress-fill");
const progressText = document.getElementById("progress-text");
const summaryEl = document.getElementById("eval-summary");
const rowsSection = document.getElementById("eval-rows");
const rowBody = document.getElementById("row-body");
const rowsEmpty = document.getElementById("rows-empty");

// The text box is the input; a chosen or dropped file is loaded into it.
let sourceName = null;
let rows = [];
let total = 0;
let filter = "all";
let controller = null;
let renderQueued = false;

function updateRunButton() {
  if (!controller) runBtn.disabled = !csvText.value.trim();
}

async function loadFile(file) {
  if (!file) return;
  csvText.value = await file.text();
  sourceName = file.name;
  fileNameEl.textContent = `Loaded ${file.name} (${(file.size / 1024).toFixed(1)} KB). You can edit it below.`;
  evalError.textContent = "";
  updateRunButton();
}

document.getElementById("choose-file").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  loadFile(fileInput.files[0]);
  fileInput.value = "";
});
document.getElementById("load-sample").addEventListener("click", async () => {
  csvText.value = await (await fetch("/static/sample.csv")).text();
  sourceName = "sample.csv";
  fileNameEl.textContent = "Loaded the sample.";
  updateRunButton();
});
document.getElementById("clear-csv").addEventListener("click", () => {
  csvText.value = "";
  sourceName = null;
  fileNameEl.textContent = "Choose a file, drop one here, or paste below.";
  updateRunButton();
  csvText.focus();
});
csvText.addEventListener("input", updateRunButton);
csvText.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runEvaluation();
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dropzone--over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dropzone--over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dropzone--over");
  loadFile(e.dataTransfer.files[0]);
});

function fmtPct(x) {
  return x == null ? "–" : (x * 100).toFixed(1) + "%";
}

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function resultLabel(r) {
  if (r.error) return "error";
  return r.correct ? "right" : "wrong";
}

function visibleRows() {
  const shown = rows.filter((r) =>
    filter === "all" ? true : filter === "wrong" ? r.correct === false : r.error != null
  );
  // Highest confidence first; rows that errored go last.
  return shown.sort((a, b) => (b.confidence ?? -1) - (a.confidence ?? -1) || a.row_number - b.row_number);
}

function renderRows() {
  renderQueued = false;
  const shown = visibleRows();
  rowBody.replaceChildren(
    ...shown.map((r) => {
      const tr = document.createElement("tr");
      tr.className = "row--" + resultLabel(r);
      const ctx = cell(r.context, "clip");
      ctx.title = r.context;
      const pred = r.error ? cell(r.error, "error") : cell(r.prediction);
      if (r.error) pred.colSpan = 3;
      tr.append(cell(r.row_number, "num"), cell(r.question), ctx, cell(r.answer), pred);
      if (!r.error) tr.append(cell(fmtPct(r.confidence), "num"), cell(fmtPct(r.answer_probability), "num"));
      const badge = document.createElement("span");
      badge.className = "badge badge--" + resultLabel(r);
      badge.textContent = { right: "✓ right", wrong: "✗ wrong", error: "error" }[resultLabel(r)];
      const td = document.createElement("td");
      td.append(badge);
      tr.append(td);
      return tr;
    })
  );
  rowsEmpty.hidden = shown.length > 0 || rows.length === 0;
}

function queueRender() {
  if (!renderQueued) {
    renderQueued = true;
    requestAnimationFrame(renderRows);
  }
}

function updateProgress(done = false) {
  const scored = rows.filter((r) => r.correct != null);
  const right = scored.filter((r) => r.correct).length;
  const errors = rows.length - scored.length;
  progressFill.style.width = total ? `${(rows.length / total) * 100}%` : "0";
  progressText.textContent =
    `${rows.length} / ${total} rows` +
    (scored.length ? ` · ${fmtPct(right / scored.length)} correct${done ? "" : " so far"}` : "") +
    (errors ? ` · ${errors} failed` : "");
}

function renderSummary(s) {
  document.getElementById("t-accuracy").textContent = fmtPct(s.accuracy);
  document.getElementById("t-correct").textContent = `${s.correct} / ${s.scored}`;
  document.getElementById("t-conf-right").textContent = fmtPct(s.mean_confidence_correct);
  document.getElementById("t-conf-wrong").textContent = fmtPct(s.mean_confidence_wrong);
  document.getElementById("t-errors").textContent = s.errors;

  document.getElementById("bands").replaceChildren(
    ...s.confidence_bands.map((b) => {
      const tr = document.createElement("tr");
      const range = `${Math.round(b.low * 100)}–${Math.round(b.high * 100)}%`;
      tr.append(cell(range), cell(b.total, "num"), cell(b.total ? fmtPct(b.accuracy) : "–", "num"));
      return tr;
    })
  );
  document.getElementById("per-question").replaceChildren(
    ...s.per_question.map((q) => {
      const tr = document.createElement("tr");
      tr.append(cell(q.question), cell(q.total, "num"), cell(fmtPct(q.accuracy), "num"));
      return tr;
    })
  );
  summaryEl.hidden = false;
}

function setRunning(running) {
  runBtn.disabled = running || !csvText.value.trim();
  runBtn.textContent = running ? "Running…" : "Run evaluation";
  cancelBtn.hidden = !running;
}

async function runEvaluation() {
  const body = csvText.value;
  if (!body.trim() || controller) return;
  evalError.textContent = "";
  rows = [];
  total = 0;
  summaryEl.hidden = true;
  rowsSection.hidden = true;
  progressEl.hidden = true;
  renderRows();

  controller = new AbortController();
  setRunning(true);
  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "text/csv" },
      body,
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || res.statusText);
    }

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const msg = JSON.parse(line);
        if (msg.type === "start") {
          total = msg.total;
          progressEl.hidden = false;
          rowsSection.hidden = false;
          updateProgress();
        } else if (msg.type === "row") {
          rows.push(msg);
          updateProgress();
          queueRender();
        } else if (msg.type === "summary") {
          updateProgress(true);
          renderSummary(msg);
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") {
      progressText.textContent = `Cancelled after ${rows.length} / ${total} rows.`;
    } else {
      evalError.textContent = e.message;
    }
  } finally {
    controller = null;
    setRunning(false);
    queueRender();
  }
}

function csvEscape(v) {
  const s = v == null ? "" : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function downloadResults() {
  const header = ["row", "context", "question", "options", "answer", "prediction",
    "confidence", "answer_probability", "correct", "error"];
  const sorted = [...rows].sort((a, b) => a.row_number - b.row_number);
  const lines = [header.join(",")].concat(
    sorted.map((r) =>
      [r.row_number, r.context, r.question, r.options.join("|"), r.answer, r.prediction,
        r.confidence, r.answer_probability, r.correct, r.error].map(csvEscape).join(",")
    )
  );
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (sourceName?.replace(/\.(csv|tsv)$/i, "") || "clef") + "-results.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

runBtn.addEventListener("click", runEvaluation);
cancelBtn.addEventListener("click", () => controller?.abort());
document.getElementById("download-results").addEventListener("click", downloadResults);
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    filter = chip.dataset.filter;
    document.querySelectorAll(".chip").forEach((c) =>
      c.setAttribute("aria-pressed", String(c === chip))
    );
    renderRows();
  })
);
