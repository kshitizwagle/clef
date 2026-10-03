const questionsEl = document.getElementById("questions");
const questionTpl = document.getElementById("question-tpl");
const answerTpl = document.getElementById("answer-tpl");
const stateEl = document.getElementById("state");
const predictBtn = document.getElementById("predict");
const formError = document.getElementById("form-error");
const resultsEl = document.getElementById("results");
const resultsEmpty = document.getElementById("results-empty");
const resultsMeta = document.getElementById("results-meta");
const statusEl = document.getElementById("status");

function addAnswer(question, text = "", focus = false) {
  const node = answerTpl.content.firstElementChild.cloneNode(true);
  const input = node.querySelector(".answer__text");
  input.value = text;
  // Enter moves to the next answer, adding one at the end, so lists can be
  // typed quickly.
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const next = node.nextElementSibling;
      if (next) next.querySelector(".answer__text").focus();
      else addAnswer(question, "", true);
    }
  });
  node.querySelector(".answer__remove").addEventListener("click", () => node.remove());
  question.querySelector(".answers").appendChild(node);
  if (focus) input.focus();
}

function addQuestion(text = "", answers = ["", ""]) {
  const node = questionTpl.content.firstElementChild.cloneNode(true);
  node.querySelector(".question__text").value = text;
  node.querySelector(".question__remove").addEventListener("click", () => node.remove());
  node.querySelector(".add-answer").addEventListener("click", () => addAnswer(node, "", true));
  answers.forEach((a) => addAnswer(node, a));
  questionsEl.appendChild(node);
  return node;
}

function collect() {
  const questions = [...questionsEl.querySelectorAll(".question")].map((q) => ({
    question: q.querySelector(".question__text").value.trim(),
    answers: [...q.querySelectorAll(".answer__text")].map((a) => a.value.trim()).filter(Boolean),
  }));
  return { state: stateEl.value.trim(), questions };
}

function validate({ state, questions }) {
  if (!state) return "Enter a state.";
  if (questions.length === 0) return "Add at least one question.";
  for (const [i, q] of questions.entries()) {
    if (!q.question) return `Question ${i + 1} is empty.`;
    if (q.answers.length < 2) return `Question ${i + 1} needs at least 2 answers.`;
    if (new Set(q.answers).size !== q.answers.length) return `Question ${i + 1} has duplicate answers.`;
  }
  return null;
}

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function renderResults(data) {
  resultsEl.replaceChildren();
  resultsEmpty.hidden = data.results.length > 0;
  resultsMeta.textContent = data.input_tokens != null ? `${data.input_tokens} input tokens` : "";

  for (const r of data.results) {
    const li = document.createElement("li");
    li.className = "result";

    const head = document.createElement("div");
    head.className = "result__head";
    const q = document.createElement("div");
    q.className = "result__question";
    q.textContent = r.question;
    const conf = document.createElement("div");
    conf.className = "result__confidence";
    conf.append("confidence ");
    const confVal = document.createElement("strong");
    confVal.textContent = pct(r.confidence);
    conf.append(confVal);
    head.append(q, conf);

    const pred = document.createElement("div");
    pred.className = "result__prediction";
    pred.append("Prediction: ");
    const predVal = document.createElement("strong");
    predVal.textContent = r.prediction;
    pred.append(predVal);

    li.append(head, pred);

    r.answers.forEach((a, i) => {
      const row = document.createElement("div");
      row.className = "prob" + (i === 0 ? " prob--top" : "");
      const bar = document.createElement("div");
      bar.className = "prob__bar";
      const fill = document.createElement("div");
      fill.className = "prob__fill";
      fill.style.width = pct(a.probability);
      const label = document.createElement("span");
      label.className = "prob__label";
      label.textContent = a.answer;
      bar.append(fill, label);
      const value = document.createElement("div");
      value.className = "prob__value";
      value.textContent = pct(a.probability);
      row.append(bar, value);
      li.append(row);
    });

    resultsEl.append(li);
  }
}

async function predict() {
  const payload = collect();
  const problem = validate(payload);
  formError.textContent = problem || "";
  if (problem) return;

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting…";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(body.detail)
        ? body.detail.map((d) => d.msg).join("\n")
        : body.detail || res.statusText;
      throw new Error(detail);
    }
    renderResults(body);
  } catch (e) {
    formError.textContent = e.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
}

async function checkStatus() {
  try {
    const res = await fetch("/api/health");
    const body = await res.json();
    statusEl.className = "status " + (body.upstream_ok ? "status--ok" : "status--down");
    statusEl.textContent = body.upstream_ok ? "model ready" : "model server down";
    statusEl.title = body.upstream;
  } catch {
    statusEl.className = "status status--down";
    statusEl.textContent = "API unreachable";
  }
}

document.getElementById("add-question").addEventListener("click", () => {
  addQuestion().querySelector(".question__text").focus();
});
predictBtn.addEventListener("click", predict);
stateEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) predict();
});

// Starting example, the same case as the smoke test.
stateEl.value = "Checkout has been failing for every customer for the last hour.";
addQuestion("Which team should handle this?", ["billing", "technical", "shipping"]);
addQuestion("Is a service down?", ["yes", "no"]);

checkStatus();
setInterval(checkStatus, 10000);

// Tabs. The selected tab is kept in the URL hash, so a reload stays on it.
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    const selected = t.dataset.view === name;
    t.setAttribute("aria-selected", String(selected));
    document.getElementById("view-" + t.dataset.view).hidden = !selected;
  });
}
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    showView(t.dataset.view);
    history.replaceState(null, "", "#" + t.dataset.view);
  })
);
showView(location.hash === "#evaluate" ? "evaluate" : "ask");
