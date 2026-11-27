"use strict";
const $ = (id) => document.getElementById(id);
const labels = [
  "Angry",
  "Disgust",
  "Fear",
  "Happy",
  "Sad",
  "Surprise",
  "Neutral",
];
const state = {
  mode: "upload",
  stream: null,
  request: null,
  generation: 0,
  result: null,
  selected: 0,
  objectURL: null,
  timer: null,
  model: null,
  busy: false,
  cameraPendingToken: null,
};
const context = $("overlay").getContext("2d");
labels.forEach((label) => {
  const row = document.createElement("div");
  row.className = "score-row";
  row.dataset.label = label;
  const name = document.createElement("span");
  name.textContent = label;
  const track = document.createElement("div");
  track.className = "score-track";
  const bar = document.createElement("div");
  bar.className = "score-fill";
  track.append(bar);
  const score = document.createElement("span");
  score.textContent = "—";
  row.append(name, track, score);
  $("score-list").append(row);
});
function feedback(message = "") {
  $("feedback").textContent = message;
}
function cancelRequest() {
  state.generation++;
  state.request?.abort();
  state.request = null;
  state.busy = false;
  clearTimeout(state.timer);
}
function resetAnalysis() {
  state.result = null;
  state.selected = 0;
  $("result-label").textContent = "Awaiting a face";
  $("result-score").replaceChildren(document.createTextNode("—"));
  const small = document.createElement("small");
  small.textContent = "MODEL SCORE";
  $("result-score").append(small);
  $("result-caption").textContent =
    "Your observation will appear here. The scores describe the model’s prediction, not how someone feels.";
  $("face-count").textContent = "—";
  $("latency").textContent = "—";
  $("analysis-state").textContent = "STANDBY";
  $("export-result").disabled = true;
  $("face-picker").hidden = true;
  document.querySelectorAll(".score-row").forEach((row) => {
    row.classList.remove("leading");
    row.querySelector(".score-fill").style.width = "0%";
    row.lastElementChild.textContent = "—";
  });
  context.clearRect(0, 0, $("overlay").width, $("overlay").height);
}
function stopCamera() {
  clearTimeout(state.timer);
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  $("camera").srcObject = null;
  $("stop-camera").hidden = true;
  $("stage-indicator").textContent = "CAMERA OFF";
}
function clearInput() {
  cancelRequest();
  stopCamera();
  if (state.objectURL) {
    URL.revokeObjectURL(state.objectURL);
    state.objectURL = null;
  }
  $("preview").removeAttribute("src");
  $("image-file").value = "";
  $("media-frame").hidden = true;
  $("empty-stage").hidden = state.mode !== "upload";
  $("camera-empty").hidden = state.mode !== "camera";
  $("change-image").hidden = true;
  $("clear-input").hidden = true;
  $("file-name").textContent =
    state.mode === "upload"
      ? "Make room for a new perspective."
      : "The camera starts only when you choose.";
  $("stage-indicator").textContent = "WAITING FOR INPUT";
  $("frame-details").textContent = "FACE DETECTION + CUSTOM CNN";
  resetAnalysis();
  feedback();
}
function mode(next) {
  if (next === state.mode) return;
  state.mode = next;
  clearInput();
  for (const item of ["upload", "camera"]) {
    $(item + "-tab").setAttribute("aria-selected", String(item === next));
    $(item + "-tab").tabIndex = item === next ? 0 : -1;
  }
  $("stage-panel").setAttribute("aria-labelledby", next + "-tab");
  $("source-note").textContent =
    next === "camera"
      ? "A NEW PERSPECTIVE, FRAME BY FRAME."
      : "ONE IMAGE. SEVEN POSSIBILITIES.";
}
$("upload-tab").addEventListener("click", () => mode("upload"));
$("camera-tab").addEventListener("click", () => mode("camera"));
for (const id of ["upload-tab", "camera-tab"])
  $(id).addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight"].includes(event.key)) {
      event.preventDefault();
      mode(state.mode === "upload" ? "camera" : "upload");
      $(state.mode + "-tab").focus();
    }
  });
for (const id of ["choose-image", "change-image"])
  $(id).addEventListener("click", () => $("image-file").click());
$("clear-input").addEventListener("click", clearInput);
function draw() {
  const canvas = $("overlay"),
    frame = $("media-frame");
  canvas.width = frame.clientWidth;
  canvas.height = frame.clientHeight;
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.result) return;
  const scale = Math.min(
      canvas.width / state.result.width,
      canvas.height / state.result.height,
    ),
    ox = (canvas.width - state.result.width * scale) / 2,
    oy = (canvas.height - state.result.height * scale) / 2;
  state.result.faces.forEach((face, index) => {
    const b = face.box,
      x = ox + b.x * scale,
      y = oy + b.y * scale,
      w = b.width * scale,
      h = b.height * scale;
    context.strokeStyle = index === state.selected ? "#ffb47f" : "#709cffaa";
    context.lineWidth = index === state.selected ? 2 : 1;
    context.strokeRect(x, y, w, h);
    const text = `${face.id} · ${face.label || "Face detected"}`;
    context.font = "12px sans-serif";
    const tw = context.measureText(text).width + 16;
    context.fillStyle = "#101b30ed";
    context.fillRect(x, Math.max(0, y - 25), tw, 23);
    context.fillStyle = "#f3f6ff";
    context.fillText(text, x + 8, Math.max(15, y - 9));
  });
}
new ResizeObserver(draw).observe($("media-frame"));
function render() {
  const result = state.result;
  if (!result) return;
  $("face-count").textContent = result.faces.length;
  $("latency").textContent = `${Math.round(result.inference_ms)} ms`;
  $("export-result").disabled = false;
  $("analysis-state").textContent = result.faces.length
    ? "OBSERVATION READY"
    : "NO FACE FOUND";
  const picker = $("face-picker");
  picker.replaceChildren();
  picker.hidden = result.faces.length < 2;
  result.faces.forEach((face, index) => {
    const b = document.createElement("button");
    b.textContent = `Face ${face.id}`;
    b.setAttribute("aria-pressed", String(state.selected === index));
    b.addEventListener("click", () => {
      state.selected = index;
      render();
    });
    picker.append(b);
  });
  const face = result.faces[state.selected];
  $("result-label").textContent =
    face?.label || (face ? "Face detected" : "No face detected");
  $("result-score").replaceChildren(
    document.createTextNode(
      face?.score != null ? `${Math.round(face.score * 100)}%` : "—",
    ),
  );
  const sm = document.createElement("small");
  sm.textContent = "MODEL SCORE";
  $("result-score").append(sm);
  $("result-caption").textContent = face?.label
    ? "The highest-scoring expression for this face. Compare the full distribution below."
    : face
      ? "Face detection is working. Expression scores will appear when your trained CNN is available."
      : "Try a front-facing portrait with clear lighting and an unobstructed face.";
  const qualityNotes = {
    small_face: "Move closer or use a higher-resolution portrait.",
    low_light: "More even lighting may improve this observation.",
    soft_focus: "A sharper portrait may improve this observation.",
  };
  const warnings = (face?.quality_flags || [])
    .map((key) => qualityNotes[key])
    .filter(Boolean);
  if (warnings.length)
    $("result-caption").textContent += " " + warnings.join(" ");
  document.querySelectorAll(".score-row").forEach((row) => {
    const score = face?.scores?.[row.dataset.label];
    row.querySelector(".score-fill").style.width =
      score != null ? `${score * 100}%` : "0%";
    row.lastElementChild.textContent =
      score != null ? `${Math.round(score * 100)}%` : "—";
    row.classList.toggle("leading", row.dataset.label === face?.label);
  });
  $("stage-indicator").textContent = state.stream
    ? "CAMERA LIVE"
    : `${result.faces.length} ${result.faces.length === 1 ? "FACE" : "FACES"} DETECTED`;
  $("frame-details").textContent =
    `${result.width} × ${result.height} / ${Math.round(result.inference_ms)} MS`;
  draw();
}
async function analyze(blob, token) {
  const controller = new AbortController();
  state.request = controller;
  state.busy = true;
  $("analysis-state").textContent = "OBSERVING";
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": blob.type },
      body: blob,
      signal: controller.signal,
    });
    const result = await response.json();
    if (token !== state.generation) return;
    if (!response.ok)
      throw new Error(
        result.detail ||
          "The analysis could not be completed. Please try again.",
      );
    state.result = result;
    state.selected = Math.min(
      state.selected,
      Math.max(0, result.faces.length - 1),
    );
    render();
    feedback(result.message || "");
  } catch (error) {
    if (token === state.generation) {
      feedback(
        error.name === "AbortError"
          ? "Analysis timed out. Please try again."
          : error.message,
      );
      $("analysis-state").textContent = "TRY AGAIN";
    }
  } finally {
    clearTimeout(timeout);
    if (token === state.generation) {
      state.busy = false;
      state.request = null;
    }
  }
}
async function selectFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    feedback("Choose a JPEG, PNG or WebP image.");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    feedback("This image is too large. Choose a file under 8 MB.");
    return;
  }
  if (state.mode !== "upload") mode("upload");
  clearInput();
  const token = state.generation;
  state.objectURL = URL.createObjectURL(file);
  $("preview").src = state.objectURL;
  $("preview").hidden = false;
  $("camera").hidden = true;
  $("empty-stage").hidden = true;
  $("media-frame").hidden = false;
  $("change-image").hidden = false;
  $("clear-input").hidden = false;
  $("file-name").textContent = file.name;
  $("stage-indicator").textContent = "ANALYSING PORTRAIT";
  $("preview").onerror = () => {
    if (token === state.generation)
      feedback("This image could not be opened. Choose another file.");
  };
  await analyze(file, token);
}
$("image-file").addEventListener("change", (event) =>
  selectFile(event.target.files[0]),
);
for (const name of ["dragenter", "dragover"])
  $("stage-panel").addEventListener(name, (event) => {
    event.preventDefault();
    if (state.mode === "upload") $("stage-panel").classList.add("dragging");
  });
for (const name of ["dragleave", "drop"])
  $("stage-panel").addEventListener(name, (event) => {
    event.preventDefault();
    $("stage-panel").classList.remove("dragging");
    if (name === "drop") selectFile(event.dataTransfer.files[0]);
  });
async function cameraFrame(token) {
  if (!state.stream || token !== state.generation) return;
  if (document.hidden) {
    state.timer = setTimeout(() => cameraFrame(token), 700);
    return;
  }
  const video = $("camera");
  if (video.readyState >= 2) {
    const canvas = document.createElement("canvas");
    const ratio = Math.min(1, 800 / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * ratio);
    canvas.height = Math.round(video.videoHeight * ratio);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.85),
    );
    if (blob && state.stream && token === state.generation)
      await analyze(blob, token);
  }
  if (state.stream && token === state.generation)
    state.timer = setTimeout(() => cameraFrame(token), 500);
}
$("start-camera").addEventListener("click", async () => {
  if (!navigator.mediaDevices?.getUserMedia) {
    feedback(
      "Camera access requires HTTPS or localhost. Try uploading a portrait instead.",
    );
    return;
  }
  cancelRequest();
  const token = state.generation;
  state.cameraPendingToken = token;
  $("start-camera").disabled = true;
  $("stop-camera").hidden = false;
  $("stop-camera").textContent = "Cancel camera";
  feedback();
  let permissionTimer;
  try {
    const capture = navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 800 },
        height: { ideal: 600 },
      },
      audio: false,
    });
    capture.then(
      (stream) => {
        if (token !== state.generation)
          stream.getTracks().forEach((t) => t.stop());
      },
      () => {},
    );
    const stream = await Promise.race([
      capture,
      new Promise((_, reject) => {
        permissionTimer = setTimeout(
          () =>
            reject(
              new DOMException(
                "Camera permission did not complete.",
                "TimeoutError",
              ),
            ),
          20000,
        );
      }),
    ]);
    clearTimeout(permissionTimer);
    if (token !== state.generation) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    state.stream = stream;
    $("camera").srcObject = stream;
    await $("camera").play();
    $("preview").hidden = true;
    $("camera").hidden = false;
    $("camera-empty").hidden = true;
    $("media-frame").hidden = false;
    $("stop-camera").hidden = false;
    $("stop-camera").textContent = "Stop camera";
    $("clear-input").hidden = false;
    $("stage-indicator").textContent = "CAMERA LIVE";
    $("file-name").textContent = "Live camera · microphone is off";
    cameraFrame(token);
  } catch (error) {
    if (token === state.generation) {
      if (error.name === "TimeoutError") cancelRequest();
      feedback(
        error.name === "TimeoutError"
          ? "Camera permission did not complete. Check your browser permissions, or upload a portrait."
          : error.name === "NotAllowedError"
            ? "Camera permission was declined. Allow it in your browser, or upload a portrait."
            : error.name === "NotFoundError"
              ? "No camera was found. Try uploading a portrait."
              : "The camera could not start. Close any other app using it and try again.",
      );
    }
  } finally {
    clearTimeout(permissionTimer);
    if (state.cameraPendingToken === token) {
      $("start-camera").disabled = false;
      if (!state.stream) $("stop-camera").hidden = true;
    }
  }
});
$("stop-camera").addEventListener("click", () => {
  cancelRequest();
  stopCamera();
  $("camera-empty").hidden = false;
  $("media-frame").hidden = true;
  resetAnalysis();
  $("file-name").textContent = "Camera stopped. Ready when you are.";
  $("frame-details").textContent = "FACE DETECTION + CUSTOM CNN";
  feedback("Camera stopped.");
});
addEventListener("pagehide", () => {
  cancelRequest();
  stopCamera();
});
$("export-result").addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob(
    [
      JSON.stringify(
        {
          ...state.result,
          observed_at: new Date().toISOString(),
          model: state.model,
          source: state.mode,
        },
        null,
        2,
      ),
    ],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "expression-observation.json";
  document.body.append(link);
  link.click();
  link.remove();
  feedback("Observation JSON prepared for download. It contains scores and face coordinates, not image pixels.");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("camera-tab").tabIndex = -1;
const dialog = $("model-dialog");
$("model-open").addEventListener("click", () => dialog.showModal());
$("model-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) {
    const r = dialog.getBoundingClientRect();
    if (
      event.clientX < r.left ||
      event.clientX > r.right ||
      event.clientY < r.top ||
      event.clientY > r.bottom
    )
      dialog.close();
  }
});
async function loadModel() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Unavailable");
    const data = await response.json();
    state.model = data.model;
    const model = data.model;
    $("model-description").textContent =
      `This project uses Nilay’s own ${model.name}, trained from random weights on ${model.dataset}. No pretrained emotion-classification weights are used. The face detector locates regions before classification.`;
    $("model-status").classList.toggle("ready", model.ready);
    $("model-status").replaceChildren(
      document.createElement("i"),
      document.createTextNode(
        model.ready
          ? model.name + " ready"
          : "Custom CNN · awaiting trained weights",
      ),
    );
    $("model-details").replaceChildren();
    const entries = {
      Model: model.name,
      Dataset: model.dataset,
      Initialization: "Random weights / from scratch",
      Input: `${model.input_size.join(" × ")} grayscale`,
      "Expression classes": model.labels.length,
      "Held-out accuracy":
        model.test_accuracy != null
          ? `${(model.test_accuracy * 100).toFixed(1)}%`
          : "Training not yet evaluated",
      "Macro F1": model.macro_f1 != null ? model.macro_f1.toFixed(3) : "—",
      "Test images": model.test_examples ?? "—",
      "Evaluation coverage": model.test_examples
        ? `${(model.test_coverage * 100).toFixed(1)}%`
        : "—",
      "Image storage": "None",
    };
    Object.entries(entries).forEach(([key, value]) => {
      const row = document.createElement("div"),
        dt = document.createElement("dt"),
        dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      row.append(dt, dd);
      $("model-details").append(row);
    });
    if (!model.ready)
      feedback(
        "Face detection is available. Expression classification will use your CNN once training is complete.",
      );
  } catch {
    $("model-status").textContent = "Connection unavailable";
    feedback(
      "The analysis service is unavailable. Please refresh in a moment.",
    );
  }
}
loadModel();
loadEvaluation();

$("sample-portrait").addEventListener("click", async () => {
  try {
    const r = await fetch("/static/assets/sample-portrait.jpg");
    if (!r.ok) throw new Error();
    selectFile(
      new File([await r.blob()], "NASA sample portrait.jpg", {
        type: "image/jpeg",
      }),
    );
  } catch {
    feedback("The sample could not be loaded. Try uploading a portrait.");
  }
});

async function loadEvaluation() {
  try {
    const response = await fetch("/api/evaluation");
    if (!response.ok) return;
    const data = await response.json(),
      metrics = data.metrics;
    if (
      !data.model.ready ||
      !metrics?.classification_report ||
      !metrics?.confusion_matrix
    )
      return;
    $("evaluation").hidden = false;
    $("evaluation-protocol").textContent =
      metrics.test_protocol ||
      "Held-out FER-2013 images. No test-set training.";
    $("evaluation-accuracy").textContent =
      `${(metrics.test_accuracy * 100).toFixed(1)}%`;
    $("evaluation-f1").textContent = metrics.macro_f1.toFixed(3);
    $("evaluation-count").textContent = metrics.test_examples.toLocaleString();
    $("evaluation-coverage").textContent =
      `${((metrics.test_coverage ?? 1) * 100).toFixed(1)}%`;
    const matrix = $("confusion-matrix");
    matrix.replaceChildren();
    function cell(parent, tag, text) {
      const c = document.createElement(tag);
      c.textContent = text;
      parent.append(c);
      return c;
    }
    const header = document.createElement("tr");
    cell(header, "th", "REF / PRED");
    labels.forEach((label) => cell(header, "th", label));
    matrix.append(header);
    metrics.confusion_matrix.forEach((row, i) => {
      const tr = document.createElement("tr");
      cell(tr, "th", labels[i]);
      const total = row.reduce((a, b) => a + b, 0);
      row.forEach((count, j) => {
        const td = cell(tr, "td", count);
        td.title = `${labels[i]} → ${labels[j]}: ${count} images`;
        td.style.backgroundColor = `rgba(121,162,255,${0.04 + (0.65 * count) / Math.max(total, 1)})`;
        if (i === j) td.className = "matrix-diagonal";
      });
      matrix.append(tr);
    });
    const report = $("class-report");
    report.replaceChildren();
    const rh = document.createElement("tr");
    ["EXPRESSION", "PRECISION", "RECALL", "F1"].forEach((text) =>
      cell(rh, "th", text),
    );
    report.append(rh);
    labels.forEach((label) => {
      const values = metrics.classification_report[label];
      if (!values) return;
      const tr = document.createElement("tr");
      cell(tr, "th", label);
      ["precision", "recall", "f1-score"].forEach((key) =>
        cell(tr, "td", values[key].toFixed(3)),
      );
      report.append(tr);
    });
    const comparison = data.comparison;
    if (
      comparison?.baseline_original_labels &&
      comparison?.improved_original_labels
    ) {
      $("comparison-panel").hidden = false;
      const table = $("model-comparison");
      table.replaceChildren();
      const h = document.createElement("tr");
      [
        "METHOD",
        "ORIGINAL LABELS · ACCURACY",
        "CROWD LABELS · ACCURACY",
        "CROWD LABELS · MACRO F1",
      ].forEach((text) => cell(h, "th", text));
      table.append(h);
      for (const [name, raw, crowd] of [
        [
          "Baseline CNN",
          comparison.baseline_original_labels,
          comparison.baseline_crowd_labels,
        ],
        [
          "Residual-SE + FERPlus",
          comparison.improved_original_labels,
          comparison.improved_crowd_labels,
        ],
      ]) {
        const tr = document.createElement("tr");
        cell(tr, "th", name);
        cell(tr, "td", `${(raw.accuracy * 100).toFixed(1)}%`);
        cell(tr, "td", `${(crowd.accuracy * 100).toFixed(1)}%`);
        cell(tr, "td", crowd.macro_f1.toFixed(3));
        table.append(tr);
      }
    }
    if (data.data_audit?.training_images) {
      const audit = data.data_audit;
      $("evaluation-audit").textContent =
        `Data audit: ${audit.training_images.toLocaleString()} training images · ${audit.validation_images.toLocaleString()} validation images · ${audit.training_overlap_removed} train/held-out overlaps removed · ${audit.training_duplicates_merged} duplicate training images merged · zero exact-duplicate leakage after cleaning. Test-time horizontal flipping is ${metrics.test_time_augmentation ? "enabled" : "disabled"}, selected using validation data only.`;
    }
  } catch {
    /* Evaluation is supplemental; the observation studio remains usable. */
  }
}
