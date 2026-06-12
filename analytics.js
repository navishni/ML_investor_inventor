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
  }));

  chart.innerHTML = metricRows.map((row) => `
    <div class="bar-row">
      <div class="bar-labels"><strong>${row.label}</strong><span>Accuracy ${row.score}% | Precision ${row.precision}% | Recall ${row.recall}% | F1 ${row.f1}% | ROC AUC ${row.roc_auc}%</span></div>
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
  }));

  matrix.innerHTML = items.map((item) => `
    <article class="matrix-card">
      <h3>${item.label}</h3>
      <div class="matrix-table">
        <div class="matrix-cell"><span>Accuracy</span><strong>${item.accuracy}%</strong></div>
        <div class="matrix-cell"><span>Precision</span><strong>${item.precision}%</strong></div>
        <div class="matrix-cell"><span>Recall</span><strong>${item.recall}%</strong></div>
        <div class="matrix-cell"><span>F1</span><strong>${item.f1}%</strong></div>
      </div>
    </article>
  `).join("");
}

document.addEventListener("DOMContentLoaded", () => {
  renderMetricBars();
  renderInsightBars();
  renderComparisonMatrix();
});
