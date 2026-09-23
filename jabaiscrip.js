// ═══════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════
const setosaData = [[1.4, 0.2], [1.4, 0.2], [1.3, 0.2], [1.5, 0.2], [1.4, 0.2], [1.7, 0.4], [1.4, 0.3], [1.5, 0.2], [1.4, 0.2], [1.5, 0.1], [1.5, 0.2], [1.6, 0.2], [1.4, 0.1], [1.1, 0.1], [1.2, 0.2], [1.5, 0.4], [1.3, 0.4], [1.4, 0.3], [1.7, 0.3], [1.5, 0.3], [1.7, 0.2], [1.5, 0.4], [1.0, 0.2], [1.7, 0.5], [1.9, 0.2], [1.6, 0.2], [1.6, 0.4], [1.5, 0.2], [1.4, 0.2], [1.6, 0.2], [1.6, 0.2], [1.5, 0.4], [1.5, 0.1], [1.4, 0.2], [1.5, 0.2], [1.2, 0.2], [1.3, 0.2], [1.4, 0.1], [1.3, 0.2], [1.5, 0.2], [1.3, 0.3], [1.3, 0.3], [1.3, 0.2], [1.6, 0.6], [1.9, 0.4], [1.4, 0.3], [1.6, 0.2], [1.4, 0.2], [1.5, 0.2], [1.4, 0.2]];
const versicolorData = [[4.7, 1.4], [4.5, 1.5], [4.9, 1.5], [4.0, 1.3], [4.6, 1.5], [4.5, 1.3], [4.7, 1.6], [3.3, 1.0], [4.6, 1.3], [3.9, 1.4], [3.5, 1.0], [4.2, 1.5], [4.0, 1.0], [4.7, 1.4], [3.6, 1.3], [4.4, 1.4], [4.5, 1.5], [4.1, 1.0], [4.5, 1.5], [3.9, 1.1], [4.8, 1.8], [4.0, 1.3], [4.9, 1.5], [4.7, 1.2], [4.3, 1.3], [4.4, 1.4], [4.8, 1.4], [5.0, 1.7], [4.5, 1.5], [3.5, 1.0], [3.8, 1.1], [3.7, 1.0], [3.9, 1.2], [5.1, 1.6], [4.5, 1.5], [4.5, 1.6], [4.7, 1.5], [4.4, 1.3], [4.1, 1.3], [4.0, 1.3], [4.4, 1.2], [4.6, 1.4], [4.0, 1.2], [3.3, 1.0], [4.2, 1.3], [4.2, 1.2], [4.2, 1.3], [4.3, 1.3], [3.0, 1.1], [4.1, 1.3]];

// ═══════════════════════════════════════════════════════════════
// MATH HELPERS
// ═══════════════════════════════════════════════════════════════
function mean(arr) {
  const s = [0, 0];
  for (const p of arr) { s[0] += p[0]; s[1] += p[1]; }
  return [s[0] / arr.length, s[1] / arr.length];
}

function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
function norm(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1]); }
function normalise(v) { const n = norm(v); return [v[0] / n, v[1] / n]; }

const mu0 = mean(setosaData);
const mu1 = mean(versicolorData);
const allData = [...setosaData, ...versicolorData];
const muAll = mean(allData);

// Pre-compute Sw
function computeSw() {
  const S = [[0, 0], [0, 0]];
  for (const p of setosaData) {
    const d0 = p[0] - mu0[0], d1 = p[1] - mu0[1];
    S[0][0] += d0 * d0; S[0][1] += d0 * d1; S[1][0] += d1 * d0; S[1][1] += d1 * d1;
  }
  for (const p of versicolorData) {
    const d0 = p[0] - mu1[0], d1 = p[1] - mu1[1];
    S[0][0] += d0 * d0; S[0][1] += d0 * d1; S[1][0] += d1 * d0; S[1][1] += d1 * d1;
  }
  return S;
}
const Sw = computeSw();

// Fisher's ratio for a given direction w
function fisherRatio(w) {
  const proj0 = dot(mu0, w);
  const proj1 = dot(mu1, w);
  const between = (proj0 - proj1) ** 2;
  let var0 = 0, var1 = 0;
  for (const p of setosaData) { const s = dot(p, w) - proj0; var0 += s * s; }
  for (const p of versicolorData) { const s = dot(p, w) - proj1; var1 += s * s; }
  const within = var0 + var1;
  return within === 0 ? 0 : between / within;
}

// Optimal LDA angle
const optW = normalise([mu1[0] - mu0[0], mu1[1] - mu0[1]]); // Simplified; real LDA uses Sw^-1
// Real optimal: eigenvector of Sw^-1 @ Sb — let's compute it
function computeOptimalW() {
  const det = Sw[0][0] * Sw[1][1] - Sw[0][1] * Sw[1][0];
  const inv = [
    [Sw[1][1] / det, -Sw[0][1] / det],
    [-Sw[1][0] / det, Sw[0][0] / det]
  ];
  // For 2-class, w* = Sw^-1 (mu1 - mu0)
  const diff = [mu1[0] - mu0[0], mu1[1] - mu0[1]];
  const w = [
    inv[0][0] * diff[0] + inv[0][1] * diff[1],
    inv[1][0] * diff[0] + inv[1][1] * diff[1]
  ];
  return normalise(w);
}
const optimalW = computeOptimalW();
const optimalAngle = Math.atan2(optimalW[1], optimalW[0]) * 180 / Math.PI;
const maxFisher = fisherRatio(optimalW);

// ═══════════════════════════════════════════════════════════════
// DOM REFS
// ═══════════════════════════════════════════════════════════════
const angleSlider = document.getElementById('angle-slider');
const plSlider = document.getElementById('pl-slider');
const pwSlider = document.getElementById('pw-slider');
const angleVal = document.getElementById('angle-val');
const plVal = document.getElementById('pl-val');
const pwVal = document.getElementById('pw-val');
const scoreVal = document.getElementById('score-val');
const threshVal = document.getElementById('thresh-val');
const predVal = document.getElementById('prediction-val');
const predBox = document.getElementById('prediction-box');
const fisherVal = document.getElementById('fisher-val');
const fisherFill = document.getElementById('fisher-fill');
const w0Val = document.getElementById('w0-val');
const w1Val = document.getElementById('w1-val');
const accVal = document.getElementById('acc-val');
const btnReset = document.getElementById('btn-reset');
const btnAnimate = document.getElementById('btn-animate');

const scatterCanvas = document.getElementById('scatterCanvas');
const projCanvas = document.getElementById('projCanvas');
const sctx = scatterCanvas.getContext('2d');
const pctx = projCanvas.getContext('2d');

// HiDPI
function setupCanvas(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = canvas.width;
  const h = canvas.height;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  ctx.scale(dpr, dpr);
  // Return logical dimensions
  return { w, h };
}

const sDim = setupCanvas(scatterCanvas, sctx);
const pDim = setupCanvas(projCanvas, pctx);

// ═══════════════════════════════════════════════════════════════
// COLORS
// ═══════════════════════════════════════════════════════════════
const COL = {
  setosa: '#5b9cf5',
  setosaFill: 'rgba(91,156,245,0.25)',
  versicolor: '#f0944c',
  versicolorFill: 'rgba(240,148,76,0.25)',
  axis: 'rgba(149,128,240,0.6)',
  axisBright: '#9580f0',
  threshold: '#e8475a',
  thresholdFill: 'rgba(232,71,90,0.1)',
  grandMean: '#e8475a',
  testPoint: '#2ac5d9',
  grid: 'rgba(255,255,255,0.04)',
  gridLabel: 'rgba(255,255,255,0.25)',
  bg: '#0f1117',
};

// ═══════════════════════════════════════════════════════════════
// SCATTER PLOT
// ═══════════════════════════════════════════════════════════════
function drawScatter(w, testPt) {
  const ctx = sctx;
  const W = sDim.w, H = sDim.h;
  const pad = { top: 20, right: 20, bottom: 45, left: 55 };
  const pw_ = W - pad.left - pad.right;
  const ph = H - pad.top - pad.bottom;

  // Data range
  const xMin = 0.3, xMax = 5.8, yMin = -0.15, yMax = 2.2;
  const toX = v => pad.left + (v - xMin) / (xMax - xMin) * pw_;
  const toY = v => pad.top + ph - (v - yMin) / (yMax - yMin) * ph;

  // Clear
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = COL.grid;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  for (let x = 1; x <= 5; x++) {
    ctx.beginPath(); ctx.moveTo(toX(x), pad.top); ctx.lineTo(toX(x), pad.top + ph); ctx.stroke();
  }
  for (let y = 0; y <= 2; y += 0.5) {
    ctx.beginPath(); ctx.moveTo(pad.left, toY(y)); ctx.lineTo(pad.left + pw_, toY(y)); ctx.stroke();
  }

  // Axes labels
  ctx.fillStyle = COL.gridLabel;
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  for (let x = 1; x <= 5; x++) ctx.fillText(x, toX(x), H - 24);
  ctx.fillText('Petal length (cm)', pad.left + pw_ / 2, H - 5);

  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let y = 0; y <= 2; y += 0.5) ctx.fillText(y.toFixed(1), pad.left - 8, toY(y));

  ctx.save();
  ctx.translate(14, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Petal width (cm)', 0, 0);
  ctx.restore();

  // ── Decision boundary (perpendicular to w through threshold point) ──
  const projMu0 = dot(mu0, w);
  const projMu1 = dot(mu1, w);
  const threshold = (projMu0 + projMu1) / 2;
  const tBound = threshold - dot(muAll, w);
  const boundPt = [muAll[0] + tBound * w[0], muAll[1] + tBound * w[1]];
  const wPerp = [-w[1], w[0]];

  // Shade the two halves lightly
  // Determine which side is setosa
  const setosaSide = projMu0 < threshold ? -1 : 1;

  // Decision boundary line
  const bp1 = [boundPt[0] - 8 * wPerp[0], boundPt[1] - 8 * wPerp[1]];
  const bp2 = [boundPt[0] + 8 * wPerp[0], boundPt[1] + 8 * wPerp[1]];
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.left, pad.top, pw_, ph);
  ctx.clip();
  ctx.strokeStyle = COL.threshold;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 6]);
  ctx.beginPath();
  ctx.moveTo(toX(bp1[0]), toY(bp1[1]));
  ctx.lineTo(toX(bp2[0]), toY(bp2[1]));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // ── LDA axis (through grand mean) ──
  const tRange = 6;
  const a1 = [muAll[0] - tRange * w[0], muAll[1] - tRange * w[1]];
  const a2 = [muAll[0] + tRange * w[0], muAll[1] + tRange * w[1]];
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad.left, pad.top, pw_, ph);
  ctx.clip();
  ctx.strokeStyle = COL.axis;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(toX(a1[0]), toY(a1[1]));
  ctx.lineTo(toX(a2[0]), toY(a2[1]));
  ctx.stroke();
  ctx.setLineDash([]);

  // ── Projection lines (drop lines from points to axis) ──
  // For the test point, draw a visible projection line
  if (testPt) {
    const tScore = dot(testPt, w);
    // Find the point on the axis closest to the test point
    const t_proj = tScore - dot(muAll, w);
    const projPt = [muAll[0] + t_proj * w[0], muAll[1] + t_proj * w[1]];
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(toX(testPt[0]), toY(testPt[1]));
    ctx.lineTo(toX(projPt[0]), toY(projPt[1]));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();

  // ── Data points ──
  function drawPoint(x, y, color, fillColor, r) {
    ctx.beginPath();
    ctx.arc(toX(x), toY(y), r, 0, Math.PI * 2);
    ctx.fillStyle = fillColor;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  for (const p of setosaData) drawPoint(p[0], p[1], COL.setosa, COL.setosaFill, 5);
  for (const p of versicolorData) drawPoint(p[0], p[1], COL.versicolor, COL.versicolorFill, 5);

  // ── Grand mean ──
  const mx = toX(muAll[0]), my = toY(muAll[1]);
  ctx.beginPath();
  const s = 7;
  ctx.moveTo(mx, my - s); ctx.lineTo(mx + s, my); ctx.lineTo(mx, my + s); ctx.lineTo(mx - s, my);
  ctx.closePath();
  ctx.fillStyle = 'rgba(244,63,94,0.6)';
  ctx.fill();
  ctx.strokeStyle = COL.grandMean;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // ── Test point ──
  if (testPt) {
    ctx.beginPath();
    ctx.arc(toX(testPt[0]), toY(testPt[1]), 7, 0, Math.PI * 2);
    ctx.fillStyle = COL.testPoint;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // ── Axis arrow label ──
  ctx.save();
  const labelPos = [muAll[0] + (tRange - 1.5) * w[0], muAll[1] + (tRange - 1.5) * w[1]];
  const lx = toX(labelPos[0]), ly = toY(labelPos[1]);
  if (lx > pad.left && lx < W - pad.right && ly > pad.top && ly < pad.top + ph) {
    ctx.fillStyle = COL.axisBright;
    ctx.font = '600 11px Inter';
    ctx.textAlign = 'left';
    ctx.fillText('w', lx + 8, ly - 6);
  }
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════════
// 1D PROJECTION — NUMBER LINE
// ═══════════════════════════════════════════════════════════════
function drawProjection(w, testPt) {
  const ctx = pctx;
  const W = pDim.w, H = pDim.h;
  const pad = { left: 50, right: 50, top: 35, bottom: 35 };
  const lineY = H / 2;
  const lineW = W - pad.left - pad.right;

  // Compute all projections
  const projS = setosaData.map(p => dot(p, w));
  const projV = versicolorData.map(p => dot(p, w));
  const allProj = [...projS, ...projV];
  const projMin = Math.min(...allProj) - 0.5;
  const projMax = Math.max(...allProj) + 0.5;

  const toX = v => pad.left + (v - projMin) / (projMax - projMin) * lineW;

  // Threshold
  const projMu0 = dot(mu0, w);
  const projMu1 = dot(mu1, w);
  const threshold = (projMu0 + projMu1) / 2;

  // Clear
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = COL.bg;
  ctx.fillRect(0, 0, W, H);

  // ── Shade regions ──
  const threshX = toX(threshold);
  const leftClass = projMu0 < projMu1 ? 'setosa' : 'versicolor';
  const rightClass = leftClass === 'setosa' ? 'versicolor' : 'setosa';

  ctx.fillStyle = leftClass === 'setosa' ? 'rgba(96,165,250,0.05)' : 'rgba(251,146,60,0.05)';
  ctx.fillRect(pad.left, pad.top - 10, threshX - pad.left, H - pad.top - pad.bottom + 20);
  ctx.fillStyle = rightClass === 'setosa' ? 'rgba(96,165,250,0.05)' : 'rgba(251,146,60,0.05)';
  ctx.fillRect(threshX, pad.top - 10, pad.left + lineW - threshX, H - pad.top - pad.bottom + 20);

  // Region labels
  ctx.font = '600 11px Inter';
  ctx.textAlign = 'center';
  ctx.fillStyle = leftClass === 'setosa' ? 'rgba(96,165,250,0.4)' : 'rgba(251,146,60,0.4)';
  ctx.fillText(leftClass.charAt(0).toUpperCase() + leftClass.slice(1), (pad.left + threshX) / 2, pad.top - 2);
  ctx.fillStyle = rightClass === 'setosa' ? 'rgba(96,165,250,0.4)' : 'rgba(251,146,60,0.4)';
  ctx.fillText(rightClass.charAt(0).toUpperCase() + rightClass.slice(1), (threshX + pad.left + lineW) / 2, pad.top - 2);

  // ── Number line ──
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pad.left, lineY);
  ctx.lineTo(pad.left + lineW, lineY);
  ctx.stroke();

  // Tick marks
  const tickStep = Math.ceil((projMax - projMin) / 8 * 2) / 2; // Round to 0.5
  const firstTick = Math.ceil(projMin / tickStep) * tickStep;
  ctx.fillStyle = COL.gridLabel;
  ctx.font = '10px JetBrains Mono';
  ctx.textAlign = 'center';
  for (let t = firstTick; t <= projMax; t += tickStep) {
    const tx = toX(t);
    ctx.beginPath();
    ctx.moveTo(tx, lineY - 5);
    ctx.lineTo(tx, lineY + 5);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillText(t.toFixed(1), tx, lineY + 20);
  }

  // ── Threshold line ──
  ctx.strokeStyle = COL.threshold;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(threshX, pad.top - 10);
  ctx.lineTo(threshX, H - pad.bottom + 10);
  ctx.stroke();
  ctx.setLineDash([]);

  // Threshold label
  ctx.fillStyle = COL.threshold;
  ctx.font = '600 10px JetBrains Mono';
  ctx.textAlign = 'center';
  ctx.fillText(`T = ${threshold.toFixed(2)}`, threshX, H - pad.bottom + 28);

  // ── Projected class means (diamond markers on the line) ──
  function drawDiamond(x, y, size, color) {
    ctx.beginPath();
    ctx.moveTo(x, y - size); ctx.lineTo(x + size, y); ctx.lineTo(x, y + size); ctx.lineTo(x - size, y);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  drawDiamond(toX(projMu0), lineY, 7, COL.setosa);
  drawDiamond(toX(projMu1), lineY, 7, COL.versicolor);

  // ── Projected data points ──
  // Use beeswarm-style offset to avoid overlap
  function beeswarm(values, radius) {
    const sorted = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
    const positions = new Array(values.length);
    for (const item of sorted) {
      const px = toX(item.v);
      let offsetY = 0;
      let placed = false;
      for (let attempt = 0; attempt < 20; attempt++) {
        const signs = attempt === 0 ? [0] : [1, -1];
        for (const sign of signs) {
          const tryY = lineY + sign * attempt * (radius * 1.1);
          let collision = false;
          for (let j = 0; j < positions.length; j++) {
            if (positions[j] === undefined) continue;
            const dx = px - positions[j].x;
            const dy = tryY - positions[j].y;
            if (Math.sqrt(dx * dx + dy * dy) < radius * 2.1) { collision = true; break; }
          }
          if (!collision) {
            offsetY = tryY;
            placed = true;
            break;
          }
        }
        if (placed) break;
      }
      if (!placed) offsetY = lineY;
      positions[item.i] = { x: px, y: offsetY };
    }
    return positions;
  }

  const r = 4.5;
  const posS = beeswarm(projS, r);
  const posV = beeswarm(projV, r);

  // Draw dots
  for (const pos of posS) {
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fillStyle = COL.setosaFill;
    ctx.fill();
    ctx.strokeStyle = COL.setosa;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
  for (const pos of posV) {
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fillStyle = COL.versicolorFill;
    ctx.fill();
    ctx.strokeStyle = COL.versicolor;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }

  // ── Test point projection ──
  if (testPt) {
    const testScore = dot(testPt, w);
    const tx = toX(testScore);
    ctx.beginPath();
    ctx.arc(tx, lineY, 6, 0, Math.PI * 2);
    ctx.fillStyle = COL.testPoint;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    // Label
    ctx.fillStyle = COL.testPoint;
    ctx.font = '600 10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(testScore.toFixed(2), tx, lineY - 18);
  }

  // Axis label
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '11px Inter';
  ctx.textAlign = 'center';
  ctx.fillText('Projection score (x · w)', pad.left + lineW / 2, H - 6);
}

// ═══════════════════════════════════════════════════════════════
// UPDATE LOOP
// ═══════════════════════════════════════════════════════════════
function update() {
  const angleDeg = parseFloat(angleSlider.value);
  const angleRad = angleDeg * Math.PI / 180;
  const w = [Math.cos(angleRad), Math.sin(angleRad)];

  const pl = parseFloat(plSlider.value);
  const pw = parseFloat(pwSlider.value);
  const testPt = [pl, pw];

  // Update display values
  angleVal.textContent = angleDeg.toFixed(1) + '°';
  plVal.textContent = pl.toFixed(2);
  pwVal.textContent = pw.toFixed(2);

  // Compute projection and classification
  const testScore = dot(testPt, w);
  const projMu0 = dot(mu0, w);
  const projMu1 = dot(mu1, w);
  const threshold = (projMu0 + projMu1) / 2;

  let prediction;
  if (projMu0 < projMu1) {
    prediction = testScore < threshold ? 'setosa' : 'versicolor';
  } else {
    prediction = testScore < threshold ? 'versicolor' : 'setosa';
  }

  scoreVal.textContent = testScore.toFixed(3);
  threshVal.textContent = threshold.toFixed(3);
  predVal.textContent = prediction.charAt(0).toUpperCase() + prediction.slice(1);
  predBox.classList.remove('setosa', 'versicolor');
  predBox.classList.add(prediction);
  predVal.style.color = prediction === 'setosa' ? COL.setosa : COL.versicolor;

  // Fisher ratio
  const jr = fisherRatio(w);
  fisherVal.textContent = jr.toFixed(3);
  const pct = Math.min(100, (jr / maxFisher) * 100);
  fisherFill.style.width = pct + '%';

  // w display
  w0Val.textContent = w[0].toFixed(4);
  w1Val.textContent = w[1].toFixed(4);

  // Accuracy
  let correct = 0;
  for (const p of setosaData) {
    const s = dot(p, w);
    const pred = projMu0 < projMu1 ? (s < threshold ? 0 : 1) : (s < threshold ? 1 : 0);
    if (pred === 0) correct++;
  }
  for (const p of versicolorData) {
    const s = dot(p, w);
    const pred = projMu0 < projMu1 ? (s < threshold ? 0 : 1) : (s < threshold ? 1 : 0);
    if (pred === 1) correct++;
  }
  accVal.textContent = `${correct}/100 (${correct}%)`;

  // Draw
  drawScatter(w, testPt);
  drawProjection(w, testPt);
}

// ═══════════════════════════════════════════════════════════════
// EVENT HANDLERS
// ═══════════════════════════════════════════════════════════════
angleSlider.addEventListener('input', update);
plSlider.addEventListener('input', update);
pwSlider.addEventListener('input', update);

// Set initial angle to optimal
angleSlider.value = optimalAngle.toFixed(1);

btnReset.addEventListener('click', () => {
  angleSlider.value = optimalAngle.toFixed(1);
  update();
});

let animating = false;
let animId = null;
btnAnimate.addEventListener('click', () => {
  if (animating) {
    animating = false;
    btnAnimate.textContent = '▶ Sweep';
    cancelAnimationFrame(animId);
    return;
  }
  animating = true;
  btnAnimate.textContent = '⏸ Stop';
  let angle = 0;
  function step() {
    if (!animating) return;
    angle += 0.4;
    if (angle > 180) angle = 0;
    angleSlider.value = angle.toFixed(1);
    update();
    animId = requestAnimationFrame(step);
  }
  step();
});

// Initial draw
update();
