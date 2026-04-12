/**
 * Face Liveness Detection — Browser Demo
 *
 * Loads model.onnx + preprocessing_meta.json, opens the webcam,
 * and runs inference on each frame using ONNX Runtime Web.
 *
 * Preprocessing mirrors the Python training pipeline exactly:
 *   resize → 224×224 → normalize with ImageNet mean/std
 */

"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let session = null;
let meta = null;
let animFrameId = null;
let running = false;

const video   = document.getElementById("video");
const overlay = document.getElementById("overlay");
const badge   = document.getElementById("prediction-badge");
const fpsEl   = document.getElementById("fps-display");
const infoEl  = document.getElementById("model-info");
const logEl   = document.getElementById("log");
const startBtn= document.getElementById("start-btn");

// Rolling FPS
const FPS_WINDOW = 30;
const fpsTimestamps = [];

// Hidden canvas for pixel extraction
const offscreen = document.createElement("canvas");
const offCtx    = offscreen.getContext("2d");


// ── Logging ────────────────────────────────────────────────────────────────
function log(msg) {
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  logEl.prepend(line);
}


// ── Model + Meta loading ───────────────────────────────────────────────────
async function loadMeta() {
  const resp = await fetch("../preprocessing_meta.json");
  if (!resp.ok) throw new Error("preprocessing_meta.json not found beside model.onnx");
  return await resp.json();
}

async function loadModel() {
  log("Loading preprocessing_meta.json…");
  meta = await loadMeta();
  log(`Meta loaded: model=${meta.model}, image_size=${meta.image_size}`);

  log("Loading model.onnx…");
  session = await ort.InferenceSession.create("../model.onnx", {
    executionProviders: ["wasm"],   // "webgl" for GPU if supported
  });
  log("Model loaded. Ready.");
  infoEl.textContent = `Model: ${meta.model} (opset ${meta.opset || "?"})`;
}


// ── Preprocessing ──────────────────────────────────────────────────────────
/**
 * Draw videoElement onto a hidden canvas, read RGBA pixels,
 * convert to a normalized Float32 tensor [1, 3, H, W].
 */
function preprocessFrame(videoEl, imageSize) {
  offscreen.width  = imageSize;
  offscreen.height = imageSize;
  offCtx.drawImage(videoEl, 0, 0, imageSize, imageSize);

  const { data } = offCtx.getImageData(0, 0, imageSize, imageSize);

  const mean = meta.mean;   // [R, G, B]
  const std  = meta.std;

  const nPix = imageSize * imageSize;
  const tensor = new Float32Array(3 * nPix);

  for (let i = 0; i < nPix; i++) {
    const r = data[i * 4 + 0] / 255.0;
    const g = data[i * 4 + 1] / 255.0;
    const b = data[i * 4 + 2] / 255.0;

    tensor[0 * nPix + i] = (r - mean[0]) / std[0];  // channel R
    tensor[1 * nPix + i] = (g - mean[1]) / std[1];  // channel G
    tensor[2 * nPix + i] = (b - mean[2]) / std[2];  // channel B
  }

  return new ort.Tensor("float32", tensor, [1, 3, imageSize, imageSize]);
}


// ── Inference ──────────────────────────────────────────────────────────────
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

async function runInference() {
  const imageSize = meta.image_size;
  const inputTensor = preprocessFrame(video, imageSize);

  const feeds = { [meta.input_name]: inputTensor };
  const results = await session.run(feeds);
  const logits = Array.from(results[meta.output_name].data);
  const probs  = softmax(logits);

  const predIdx = probs.indexOf(Math.max(...probs));
  const label   = meta.class_map[String(predIdx)];
  const conf    = probs[predIdx];

  return { label, conf, probs };
}


// ── Overlay drawing ────────────────────────────────────────────────────────
function drawOverlay(label, conf) {
  const ctx = overlay.getContext("2d");
  const W   = overlay.width;
  const H   = overlay.height;
  ctx.clearRect(0, 0, W, H);

  const colorMap = { live: "#22dd55", spoof: "#ff3333" };
  const color = colorMap[label] || "#888888";

  // Border
  ctx.strokeStyle = color;
  ctx.lineWidth   = 4;
  ctx.strokeRect(2, 2, W - 4, H - 4);
}

function updateBadge(label, conf) {
  badge.className = label === "live" ? "live" : label === "spoof" ? "spoof" : "none";
  if (label === "live" || label === "spoof") {
    badge.textContent = `${label.toUpperCase()}  ${(conf * 100).toFixed(1)}%`;
  } else {
    badge.textContent = "—";
  }
}


// ── Rolling FPS ────────────────────────────────────────────────────────────
function tickFPS() {
  const now = performance.now();
  fpsTimestamps.push(now);
  while (fpsTimestamps.length > FPS_WINDOW) fpsTimestamps.shift();
  if (fpsTimestamps.length < 2) return;
  const elapsed = (fpsTimestamps.at(-1) - fpsTimestamps[0]) / 1000;
  const fps = (fpsTimestamps.length - 1) / elapsed;
  fpsEl.textContent = `FPS: ${fps.toFixed(1)}`;
}


// ── Main loop ──────────────────────────────────────────────────────────────
async function inferenceLoop() {
  if (!running) return;

  if (video.readyState >= 2) {  // HAVE_CURRENT_DATA
    try {
      const { label, conf } = await runInference();
      drawOverlay(label, conf);
      updateBadge(label, conf);
    } catch (err) {
      // Don't crash the loop on a single bad frame
      console.warn("Inference error:", err);
    }
    tickFPS();
  }

  animFrameId = requestAnimationFrame(inferenceLoop);
}


// ── Webcam setup ───────────────────────────────────────────────────────────
async function startWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
  });
  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });

  // Size overlay canvas to match rendered video
  overlay.width  = video.videoWidth  || 640;
  overlay.height = video.videoHeight || 480;
  overlay.style.width  = video.offsetWidth  + "px";
  overlay.style.height = video.offsetHeight + "px";

  log("Webcam started.");
}


// ── Entry point ─────────────────────────────────────────────────────────────
async function startDemo() {
  startBtn.disabled = true;
  startBtn.textContent = "Starting…";

  try {
    await loadModel();
    await startWebcam();

    running = true;
    inferenceLoop();

    startBtn.textContent = "Running";
    log("Inference loop started.");
  } catch (err) {
    log(`ERROR: ${err.message}`);
    infoEl.textContent = "Failed to load — see log below.";
    startBtn.disabled = false;
    startBtn.textContent = "Retry";
    console.error(err);
  }
}
