/** @type {import('@tensorflow/tfjs')} */
const tf = window.tf;

const baseroute = "https://fedlearn.sweng.qzz.io" //change to "" while local
const socket = io(baseroute + "/api", { transports: ["websocket"] });
socket.on("round_countdown", data => {
  const rn = document.getElementById("roundNumber");
  const cd = document.getElementById("countdown");
  if (rn) rn.textContent = data.current_round;
  if (cd) cd.textContent = data.seconds_left;
  if (data.previous_round_status) {
    document.getElementById("prevStatus").textContent = data.previous_round_status;
  }
  if (data.model_size_kb !== undefined) {
    document.getElementById("modelSize").textContent = `${data.model_size_kb} KB`;
  }
});

socket.on("round_status", data => {
  console.log("Server status:", data.status);
  const cd = document.getElementById("countdown");
  if (cd && data.status === "updating") cd.textContent = "Updating…";
  if (data.previous_round_status) {
    document.getElementById("prevStatus").textContent = data.previous_round_status;
  }
  if (data.model_size_kb !== undefined) {
    document.getElementById("modelSize").textContent = `${data.model_size_kb} KB`;
  }
});

socket.on("round_finished", data => {
  console.log("Round finished:", data);
  const rn = document.getElementById("roundNumber");
  const cd = document.getElementById("countdown");
  if (rn) rn.textContent = data.round;
  if (cd) cd.textContent = "Evaluating…";
  refreshChart();
  if (data.previous_round_status) {
    document.getElementById("prevStatus").textContent = data.previous_round_status;
  }
  if (data.model_size_kb !== undefined) {
    document.getElementById("modelSize").textContent = `${data.model_size_kb} KB`;
  }
});

function log(msg) {
  const logDiv = document.getElementById('log');
  logDiv.textContent =
    `[${new Date().toLocaleTimeString()}] ` + msg + "\n" + logDiv.textContent;
}


(function setupCanvas() {
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineWidth = 18;
  ctx.strokeStyle = 'black';

  let drawing = false;

  function pos(e) {
    const rect = canvas.getBoundingClientRect();
    const client = e.touches ? e.touches[0] : e;
    return { x: client.clientX - rect.left, y: client.clientY - rect.top };
  }

  canvas.addEventListener('pointerdown', e => {
    drawing = true;
    const p = pos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  });

  canvas.addEventListener('pointermove', e => {
    if (!drawing) return;
    const p = pos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });

  window.addEventListener('pointerup', () => {
    drawing = false;
  });

  document.getElementById('clearBtn').onclick = () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    renderMNISTPreview();
  };
})();


function preprocessCanvasToMNIST() {
  const src = document.getElementById("drawCanvas");
  const w = src.width, h = src.height;

  const ctx = src.getContext("2d");
  const img = ctx.getImageData(0, 0, w, h);
  const data = img.data;

  const gray = new Float32Array(w * h);
  for (let i = 0; i < gray.length; i++) {
    const r = data[i * 4];
    gray[i] = 255 - r;
  }

  const th = 30;
  for (let i = 0; i < gray.length; i++) {
    if (gray[i] < th) gray[i] = 0;
  }

  let minX = w, minY = h, maxX = 0, maxY = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = gray[y * w + x];
      if (v > 0) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (minX >= maxX || minY >= maxY) {
    return new Float32Array(28 * 28);
  }

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;

  const crop = document.createElement("canvas");
  crop.width = boxW;
  crop.height = boxH;
  const cctx = crop.getContext("2d");

  const cropData = cctx.createImageData(boxW, boxH);
  for (let y = 0; y < boxH; y++) {
    for (let x = 0; x < boxW; x++) {
      const v = gray[(minY + y) * w + (minX + x)];
      const i = (y * boxW + x) * 4;
      const px = 255 - v;
      cropData.data[i] = px;
      cropData.data[i + 1] = px;
      cropData.data[i + 2] = px;
      cropData.data[i + 3] = 255;
    }
  }
  cctx.putImageData(cropData, 0, 0);

  const S = Math.max(boxW, boxH);
  const pad = document.createElement("canvas");
  pad.width = S;
  pad.height = S;
  const pctx = pad.getContext("2d");

  pctx.fillStyle = "white";
  pctx.fillRect(0, 0, S, S);

  pctx.drawImage(crop, (S - boxW) / 2, (S - boxH) / 2);

  const f = document.createElement("canvas");
  f.width = 28;
  f.height = 28;
  f.getContext("2d").drawImage(pad, 0, 0, 28, 28);

  const small = f.getContext("2d").getImageData(0, 0, 28, 28);
  const out = new Float32Array(28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const r = small.data[i * 4];
    out[i] = (255 - r) / 255;
  }

  return out;
}


function renderMNISTPreview(showGrid = true) {
  const arr = preprocessCanvasToMNIST();
  const canvas = document.getElementById("mnistPreview");
  const ctx = canvas.getContext("2d");

  const scale = canvas.width / 28;

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = Math.round((1 - arr[y * 28 + x]) * 255);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }

  if (showGrid) {
    ctx.strokeStyle = "#ddd";
    ctx.lineWidth = 0.6;
    for (let i = 0; i <= 28; i++) {
      ctx.beginPath();
      ctx.moveTo(i * scale, 0);
      ctx.lineTo(i * scale, canvas.height);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, i * scale);
      ctx.lineTo(canvas.width, i * scale);
      ctx.stroke();
    }
  }
}


let localData = [];
let feedbackQueue = [];
let feedbackTimeout = null;
const FEEDBACK_BATCH_DELAY = 3000;


function loadImageAs28x28(file, label) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const temp = document.createElement("canvas");
      temp.width = 280;
      temp.height = 280;
      const tctx = temp.getContext("2d");

      tctx.fillStyle = "white";
      tctx.fillRect(0, 0, 280, 280);
      tctx.drawImage(img, 0, 0, 280, 280);

      const main = document.getElementById("drawCanvas");
      const mctx = main.getContext("2d");
      mctx.drawImage(temp, 0, 0, 280, 280);

      const arr = preprocessCanvasToMNIST();
      localData.push({ x: arr, y: label });

      log(`Uploaded image added (label=${label}). Samples = ${localData.length}`);

      resolve();
    };
    img.src = URL.createObjectURL(file);
  });
}


function createModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [784] }));
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  m.compile({
    optimizer: tf.train.adam(0.0015),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return m;
}

async function serializeModelWeights(model) {
  const weights = model.getWeights();
  const shapes = weights.map(w => w.shape);
  const arrays = [];

  for (const w of weights) {
    arrays.push(Array.from(await w.data()));
  }
  return { shapes, weights: arrays };
}

async function loadWeightsIntoModel(model, gm) {
  const tensors = gm.weights.map(
    (arr, i) => tf.tensor(arr, gm.shapes[i])
  );
  model.setWeights(tensors);
  tensors.forEach(t => t.dispose());
}


async function fetchGlobalModel() {
  return await fetch(`${baseroute}/api/get_model`).then(r => r.json());
}

async function predictDrawnDigit() {
  try {
    console.log("predictDrawnDigit called");

    const gm = await fetchGlobalModel();
    if (!gm.initialized) {
      document.getElementById('prediction').textContent =
        "Global model not initialized";
      return;
    }

    const model = createModel();
    await loadWeightsIntoModel(model, gm);

    const arr = preprocessCanvasToMNIST();
    const xs = tf.tensor2d([arr], [1, 784]);

    const pred = model.predict(xs);
    const predData = await pred.data();

    let maxProb = -1;
    let idx = 0;
    for (let i = 0; i < predData.length; i++) {
      if (predData[i] > maxProb) {
        maxProb = predData[i];
        idx = i;
      }
    }

    document.getElementById('prediction').textContent =
      `Predicted: ${idx} (${(maxProb * 100).toFixed(2)}%)`;
    document.getElementById('lastPredictionText').textContent = idx;

    const barCanvas = document.getElementById('predictionBars');
    const ctx = barCanvas.getContext('2d');
    ctx.clearRect(0, 0, barCanvas.width, barCanvas.height);

    const leftMargin = 40;
    const rightMargin = 10;
    const topMargin = 10;
    const bottomMargin = 10;
    const n = predData.length;

    const availableHeight = Math.max(100, barCanvas.height - topMargin - bottomMargin);
    const slot = availableHeight / n;
    const barHeight = Math.max(8, slot * 0.65);
    const gap = Math.max(2, slot * 0.35);
    const maxWidth = barCanvas.width - leftMargin - rightMargin;

    ctx.font = "12px Arial";
    ctx.textBaseline = "middle";

    for (let i = 0; i < n; i++) {
      const y = topMargin + i * (barHeight + gap);

      ctx.fillStyle = "#eee";
      ctx.fillRect(leftMargin, y, maxWidth, barHeight);

      const barWidth = predData[i] * maxWidth;
      ctx.fillStyle = i === idx ? "#ff8800" : "#0077ff";
      ctx.fillRect(leftMargin, y, barWidth, barHeight);

      ctx.fillStyle = "#000";
      ctx.textAlign = "right";
      ctx.fillText(i, leftMargin - 8, y + barHeight / 2);

      ctx.textAlign = "left";
      const probText = (predData[i] * 100).toFixed(1) + "%";
      const measured = ctx.measureText(probText).width;
      const insideX = leftMargin + Math.max(4, barWidth - measured - 4);
      const outsideX = leftMargin + barWidth + 6;
      const textX = barWidth > measured + 8 ? insideX : outsideX;

      ctx.fillStyle = barWidth > measured + 8 ? "#fff" : "#000";
      ctx.fillText(probText, textX, y + barHeight / 2);
    }


    xs.dispose();
    pred.dispose();
    model.dispose();

  } catch (err) {
    console.error("Prediction failed:", err);
    document.getElementById('prediction').textContent = "Prediction error!";
  }
}

async function submitDeltaUpdate(deltaObj, size) {
  return await fetch(`${baseroute}/api/submit_update`, {
    method: 'POST',
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      deltas: deltaObj.deltas,
      shapes: deltaObj.shapes,
      client_size: size,
      is_delta: true
    })
  }).then(r => r.json());
}


async function localTrainAndSubmit(batchData = null) {
  const dataToTrain = Array.isArray(batchData) ? batchData :
                      Array.isArray(localData) ? localData : [];

  if (!dataToTrain.length) {
    log("No samples to train.");
    return;
  }
  log(`Starting local training on ${dataToTrain.length} sample(s)`);

  const model = createModel();
  const gm = await fetchGlobalModel();
  if (gm.initialized) await loadWeightsIntoModel(model, gm);
  const before = await serializeModelWeights(model);
  const xs = tf.tensor2d(dataToTrain.map(s => s.x), [dataToTrain.length, 784]);
  const ys = tf.oneHot(tf.tensor1d(dataToTrain.map(s => s.y), 'int32'), 10);

  await model.fit(xs, ys, {
    epochs: 3,
    shuffle: true,
    batchSize: Math.min(8, dataToTrain.length),
    callbacks: {
      onEpochEnd: async (e, logs) => {
        log(`Epoch ${e+1}: loss=${logs.loss.toFixed(3)} acc=${logs.acc?.toFixed(3)}`);
        await tf.nextFrame();
      }
    }
  });

  const after = await serializeModelWeights(model);
  const deltas = after.weights.map((arr, i) =>
    arr.map((v, j) => v - before.weights[i][j])
  );
  await submitDeltaUpdate({ shapes: after.shapes, deltas }, dataToTrain.length);
  log("Submitted update to server.");

  xs.dispose();
  ys.dispose();
  model.dispose();
}



async function handleUserFeedback(label) {
  const arr = preprocessCanvasToMNIST();
  feedbackQueue.push({ x: arr, y: label });

  log(`Feedback sample added (label=${label}). Queue length=${feedbackQueue.length}`);

  if (feedbackTimeout) clearTimeout(feedbackTimeout);
  feedbackTimeout = setTimeout(async () => {
    if (feedbackQueue.length === 0) return;

    log(`Training on ${feedbackQueue.length} feedback sample(s)...`);
    await localTrainAndSubmit(feedbackQueue);
    feedbackQueue = [];
  }, FEEDBACK_BATCH_DELAY);
}

const accCanvas = document.getElementById("accuracyChart");
const accCtx = accCanvas.getContext("2d");
const userAccDisplay = document.getElementById("userAccDisplay");
const mnistAccDisplay = document.getElementById("mnistAccDisplay");

async function fetchEvaluationLogForChart() {
  try {
    const res = await fetch(`${baseroute}/api/evaluation_log?t=` + Date.now());
    if (!res.ok) return [];
    return await res.json();
  } catch (e) {
    console.warn("Failed fetching eval log:", e);
    return [];
  }
}

function downsampleSeries(values, times, maxPoints = 150) {
  const n = values.length;
  if (n <= maxPoints) return { values, times };

  const step = n / (maxPoints - 2);
  const newValues = [values[0]];
  const newTimes = [times[0]];

  for (let i = 1; i < maxPoints - 1; i++) {
    const idx = Math.floor(i * step);
    newValues.push(values[idx]);
    newTimes.push(times[idx]);
  }

  newValues.push(values[n - 1]);
  newTimes.push(times[n - 1]);

  return { values: newValues, times: newTimes };
}


async function drawCombinedAccuracyChart() {
  const evalLog = await fetchEvaluationLogForChart();

  if (!evalLog.length) {
    accCtx.clearRect(0, 0, accCanvas.width, accCanvas.height);
    accCtx.fillStyle = "#333";
    accCtx.font = "14px Arial";
    accCtx.fillText("No evaluation data yet.", 20, 40);
    return;
  }

  const rounds = evalLog.map(e => e.round);
  const modelAcc = evalLog.map(e => e.accuracy_model ?? 0);

  let allVotes = [];
  const communityCumulative = evalLog.map(e => {
    if (Array.isArray(e.community_votes) && e.community_votes.length) {
      allVotes.push(...e.community_votes);
    }
    return allVotes.length ? allVotes.reduce((a,b)=>a+b,0)/allVotes.length : 0;
  });

  const dsModel = downsampleSeries(modelAcc, rounds, 150);
  const dsCommunity = downsampleSeries(communityCumulative, rounds, 150);

  const dsRounds = dsModel.times;

  mnistAccDisplay.textContent = (modelAcc[modelAcc.length - 1] * 100).toFixed(2) + "%";
  userAccDisplay.textContent = (communityCumulative[communityCumulative.length - 1] * 100).toFixed(2) + "%";

  const w = accCanvas.width;
  const h = accCanvas.height;
  const ctx = accCtx;
  ctx.clearRect(0, 0, w, h);

  const left = 50, right = w - 20;
  const top = 20, bottom = h - 40;
  const plotW = right - left, plotH = bottom - top;

  ctx.strokeStyle = "#000";
  ctx.beginPath();
  ctx.moveTo(left, top);
  ctx.lineTo(left, bottom);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(left, bottom);
  ctx.lineTo(right, bottom);
  ctx.stroke();

  ctx.fillStyle = "#000";
  ctx.font = "12px Arial";
  ctx.fillText("100%", 12, top + 5);
  ctx.fillText("50%", 16, top + plotH / 2 + 4);
  ctx.fillText("0%", 25, bottom + 4);

  const minX = dsRounds[0];
  const maxX = dsRounds[dsRounds.length - 1] || minX + 1;

  function mapX(roundIndex) {
    if (maxX === minX) return left;
    return left + ((roundIndex - minX) / (maxX - minX)) * plotW;
  }
  function mapY(v) {
    return bottom - v * plotH;
  }

  ctx.strokeStyle = "#0077ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  dsModel.values.forEach((v, i) => {
    const x = mapX(dsRounds[i]);
    const y = mapY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.strokeStyle = "#ff8800";
  ctx.lineWidth = 2;
  ctx.beginPath();
  dsCommunity.values.forEach((v, i) => {
    const x = mapX(dsRounds[i]);
    const y = mapY(v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.font = "10px Arial";
  ctx.fillStyle = "#000";
  const nTicks = 6;
  for (let i = 0; i < nTicks; i++) {
    const r = minX + ((maxX - minX) * i) / (nTicks - 1);
    const x = mapX(r);
    ctx.fillText("R" + Math.round(r), x - 12, bottom + 14);
  }

  ctx.fillStyle = "#ff8800";
  ctx.fillRect(right - 140, top + 4, 12, 8);
  ctx.fillStyle = "#000";
  ctx.fillText("Community", right - 120, top + 12);

  ctx.fillStyle = "#0077ff";
  ctx.fillRect(right - 60, top + 4, 12, 8);
  ctx.fillStyle = "#000";
  ctx.fillText("MNIST", right - 40, top + 12);
}


async function sendFeedbackToServer(label) {
  const lastPrediction = parseInt(document.getElementById('lastPredictionText').textContent);
  if (isNaN(lastPrediction)) return;

  await handleUserFeedback(label);

  const userAcc = lastPrediction === label ? 1 : 0;

  let round = 0;
  try {
    const logRes = await fetch(`${baseroute}/api/evaluation_log?t=` + Date.now());
    const logData = await logRes.json();
    if (logData.length > 0) round = logData[logData.length - 1].round;
  } catch (err) {
    console.warn("Could not fetch current round, defaulting to 0:", err);
  }

  try {
    const res = await fetch(`${baseroute}/api/submit_user_feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        round: round,
        user_accuracy: userAcc
      })
    });
    const data = await res.json();
    log(`Feedback submitted to server: ${JSON.stringify(data)}`);
  } catch (err) {
    console.error("Failed to submit feedback to server:", err);
  }
}

function updateTimestamp() {
  const now = new Date().toLocaleString();
  document.getElementById("lastUpdated").textContent = "Last updated: " + now;
}

let drawCombinedAccuracyChartCached = (function () {
  let running = false;
  return function () {
    if (running) return;
    running = true;
    drawCombinedAccuracyChart().finally(() => { running = false; });
  };
})();

function refreshChart() {
  drawCombinedAccuracyChartCached();
  updateTimestamp();
}

// schedule main chart refresh every minute, need to sync with round_freq in main
setInterval(refreshChart, 60_000);


window.onload = () => {
  document.getElementById('saveDrawingBtn').onclick = () => {
    const label = parseInt(document.getElementById('digitLabel').value);
    const arr = preprocessCanvasToMNIST();
    localData.push({ x: arr, y: label });
    log(`Saved drawing (${label}). Total ${localData.length}`);
  };

  document.getElementById('uploadBtn').onclick = async () => {
    const f = document.getElementById('uploadImage').files[0];
    if (!f) return log("No file selected");
    await loadImageAs28x28(f, parseInt(document.getElementById('digitLabel').value));
  };

  document.getElementById('drawCanvas').addEventListener('pointerup', renderMNISTPreview);

  document.getElementById('feedbackYesBtn').onclick = async () => {
    const lastPrediction = parseInt(document.getElementById('lastPredictionText').textContent);
    if (!isNaN(lastPrediction)) await sendFeedbackToServer(lastPrediction);
  };

  document.getElementById('feedbackNoBtn').onclick = () => {
    document.getElementById('correctionUI').style.display = 'inline-block';
  };

  document.getElementById('submitCorrectionBtn').onclick = async () => {
    const correctLabel = parseInt(document.getElementById('correctLabelInput').value);
    if (!isNaN(correctLabel)) {
      document.getElementById('correctionUI').style.display = 'none';
      await sendFeedbackToServer(correctLabel);
    }
  };

  const btn = document.getElementById('predictBtn');
  if (!btn) {
    console.warn("Predict button not found!");
  }
  else btn.onclick = async (e) => {
      e.preventDefault();
      await predictDrawnDigit();
    };

  document.getElementById('trainBtn').onclick = localTrainAndSubmit;

  document.getElementById('evalBtn').onclick = async () => {
    const res = await fetch(`${baseroute}/api/evaluate_model`);
    log("Eval: " + JSON.stringify(await res.json()));
    refreshChart();
  };

  document.getElementById('showModelBtn').onclick = displayGlobalModel;

  refreshChart();
};


async function displayGlobalModel() {
  const gm = await fetchGlobalModel();
  const div = document.getElementById('modelState');

  if (!gm.initialized) {
    div.textContent = "Global model not initialized.";
    return;
  }

  let out = "";
  gm.weights.forEach((w, i) => {
    const min = Math.min(...w).toFixed(4);
    const max = Math.max(...w).toFixed(4);
    const mean = (w.reduce((a,b)=>a+b,0)/w.length).toFixed(4);
    out += `Layer ${i} (${gm.shapes[i]}) → min:${min} max:${max} mean:${mean}\n`;
  });

  div.innerHTML = out.replace(/\n/g, "<br>");
}
