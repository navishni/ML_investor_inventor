const CURVE_COLORS = ["#37d5ff", "#ffb14a", "#83d978", "#ff7085"];

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function renderMetricBars() {
  const payload = window.MATCHTANK_GRAPHS;
  const chart = document.getElementById("metricChart");
  if (!payload || !chart) return;
  const metricRows = payload.labels.map((label, index) => ({
    label,
    score: payload.accuracy[index],
    precision: payload.precision[index],
    recall: payload.recall[index],
    f1: payload.f1[index],
    roc_auc: payload.roc_auc[index],
    pr_auc: payload.pr_auc?.[index],
    decision_score: payload.decision_score?.[index],
  }));

  chart.innerHTML = metricRows.map((row) => `
    <div class="bar-row">
      <div class="bar-labels">
        <strong>${row.label}</strong>
        <span>Accuracy ${row.score}% | Precision ${row.precision}% | Recall ${row.recall}% | F1 ${row.f1}% | ROC AUC ${row.roc_auc}%${row.pr_auc != null ? ` | PR AUC ${row.pr_auc}%` : ""}${row.decision_score != null ? ` | Decision ${row.decision_score}%` : ""}</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width:${row.score}%"></div></div>
    </div>
  `).join("");
}

function renderInsightBars() {
  const insights = window.MATCHTANK_INSIGHTS;
  const chart = document.getElementById("insightChart");
  if (!insights || !chart) return;
  const rows = [...insights.top_domains, ...insights.top_technologies];
  chart.innerHTML = rows.map((row) => `
    <div class="bar-row">
      <div class="bar-labels"><strong>${row.label}</strong><span>${row.value}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, row.value)}%"></div></div>
    </div>
  `).join("");
}

function renderComparisonMatrix() {
  const payload = window.MATCHTANK_GRAPHS;
  const matrix = document.getElementById("comparisonMatrix");
  if (!payload || !matrix) return;
  const items = payload.labels.map((label, index) => ({
    label,
    accuracy: payload.accuracy[index],
    precision: payload.precision[index],
    recall: payload.recall[index],
    f1: payload.f1[index],
    roc_auc: payload.roc_auc[index],
    pr_auc: payload.pr_auc?.[index],
    decision_score: payload.decision_score?.[index],
  }));

  matrix.innerHTML = items.map((item) => `
    <article class="matrix-card">
      <h3>${item.label}</h3>
      <div class="matrix-table">
        <div class="matrix-cell"><span>Accuracy</span><strong>${item.accuracy}%</strong></div>
        <div class="matrix-cell"><span>Precision</span><strong>${item.precision}%</strong></div>
        <div class="matrix-cell"><span>Recall</span><strong>${item.recall}%</strong></div>
        <div class="matrix-cell"><span>F1</span><strong>${item.f1}%</strong></div>
        <div class="matrix-cell"><span>ROC AUC</span><strong>${item.roc_auc}%</strong></div>
        <div class="matrix-cell"><span>PR AUC</span><strong>${item.pr_auc != null ? `${item.pr_auc}%` : "n/a"}</strong></div>
        <div class="matrix-cell"><span>Decision</span><strong>${item.decision_score != null ? `${item.decision_score}%` : "n/a"}</strong></div>
      </div>
    </article>
  `).join("");
}

function scalePoint(value, min, max, start, span, inverted = false) {
  const safeMax = max === min ? min + 1 : max;
  const ratio = (toNumber(value) - min) / (safeMax - min);
  return inverted ? start + (1 - ratio) * span : start + ratio * span;
}

function buildLinePath(points) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(" ");
}

function renderCurveLegend(legendId, curves, metricKey) {
  const legend = document.getElementById(legendId);
  if (!legend) return;
  legend.innerHTML = curves.map((curve, index) => {
    const color = CURVE_COLORS[index % CURVE_COLORS.length];
    const metricValue = metricKey === "roc" ? curve.roc_auc : curve.pr_auc;
    const metricLabel = metricKey === "roc" ? "ROC AUC" : "PR AUC";
    return `
      <span class="curve-legend-item">
        <span class="curve-dot" style="background:${color};"></span>
        <span>
          <strong>${curve.name}</strong>
          <small>${metricLabel} ${metricValue}%</small>
        </span>
      </span>
    `;
  }).join("");
}

function renderCurveChart(containerId, legendId, seriesKey, options) {
  const payload = window.MATCHTANK_GRAPHS || {};
  const curves = (payload.curves || []).slice(0, 3);
  const container = document.getElementById(containerId);
  if (!container) return;

  renderCurveLegend(legendId, curves, seriesKey);

  if (!curves.length) {
    container.innerHTML = '<div class="note">No curve data available yet.</div>';
    return;
  }

  const width = 700;
  const height = 380;
  const pad = 52;
  const plotWidth = width - pad * 2;
  const plotHeight = height - pad * 2;
  const gridTicks = [0, 0.25, 0.5, 0.75, 1];
  const positiveRate = toNumber(payload.positive_rate) / 100;

  const grid = gridTicks.map((tick) => {
    const y = pad + (1 - tick) * plotHeight;
    const x = pad + tick * plotWidth;
    return `
      <line class="curve-grid-line" x1="${pad}" y1="${y}" x2="${width - pad}" y2="${y}"></line>
      <line class="curve-grid-line" x1="${x}" y1="${pad}" x2="${x}" y2="${height - pad}"></line>
      <text class="curve-axis-label" x="${pad - 10}" y="${y + 4}" text-anchor="end">${Math.round(tick * 100)}%</text>
      <text class="curve-axis-label" x="${x}" y="${height - pad + 18}" text-anchor="middle">${Math.round(tick * 100)}%</text>
    `;
  }).join("");

  const baseline = seriesKey === "roc"
    ? `<path class="curve-baseline" d="M ${pad} ${height - pad} L ${width - pad} ${pad}"></path>`
    : `<line class="curve-baseline" x1="${pad}" y1="${pad + (1 - positiveRate) * plotHeight}" x2="${width - pad}" y2="${pad + (1 - positiveRate) * plotHeight}"></line>`;

  const modelCurves = curves.map((curve, index) => {
    const series = curve[seriesKey];
    if (!series) return "";
    const xValues = seriesKey === "roc"
      ? (series.fpr || series.x || [])
      : (series.recall || series.x || []);
    const yValues = seriesKey === "roc"
      ? (series.tpr || series.y || [])
      : (series.precision || series.y || []);
    if (!xValues || !yValues) return "";
    const points = xValues.map((x, pointIndex) => ({
      x: scalePoint(x, 0, 1, pad, plotWidth),
      y: scalePoint(yValues[pointIndex], 0, 1, pad, plotHeight, true),
    }));
    const path = buildLinePath(points);
    const color = CURVE_COLORS[index % CURVE_COLORS.length];
    const endPoint = points[points.length - 1];
    return `
      <path class="curve-line" d="${path}" style="stroke:${color}"></path>
      <circle cx="${endPoint.x}" cy="${endPoint.y}" r="3.8" fill="${color}" fill-opacity="0.95"></circle>
    `;
  }).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" class="curve-svg" role="img" aria-label="${options.title}">
      <rect class="curve-bg" x="0" y="0" width="${width}" height="${height}" rx="18"></rect>
      ${grid}
      ${baseline}
      ${modelCurves}
      <text class="curve-axis-title" x="${width / 2}" y="${height - 8}" text-anchor="middle">${options.xLabel}</text>
      <text class="curve-axis-title" x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">${options.yLabel}</text>
    </svg>
  `;
}

document.addEventListener("DOMContentLoaded", () => {
  renderMetricBars();
  renderInsightBars();
  renderComparisonMatrix();
  renderCurveChart("rocChart", "rocLegend", "roc", {
    title: "ROC curve comparison",
    xLabel: "False Positive Rate",
    yLabel: "True Positive Rate",
  });
  renderCurveChart("prChart", "prLegend", "pr", {
    title: "Precision-recall curve comparison",
    xLabel: "Recall",
    yLabel: "Precision",
  });
});
