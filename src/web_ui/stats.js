const sources = ['image', 'video', 'webcam'];
const storageKey = 'seminar-trashnet-stats-v1';
const number = new Intl.NumberFormat('vi-VN');
const decimal = new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const percent = new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 1 });

function empty(labels) {
  return { uncertain: 0, warming: 0, time: 0, labels: Object.fromEntries(labels.map(label => [label, 0])) };
}

function accepted(tally) {
  return Object.values(tally.labels).reduce((sum, count) => sum + count, 0);
}

function total(tally) {
  return accepted(tally) + tally.uncertain + tally.warming;
}

export function createStatistics(config) {
  const byId = id => document.getElementById(id);
  const labelKeys = Object.keys(config.labels);
  const modelFilter = byId('stats-model');
  const sourceFilter = byId('stats-source');
  const reset = byId('stats-reset');
  const labelRows = new Map();
  const modelRows = new Map();
  let buckets = {};

  // Tab storage survives refreshes and keeps separate browser tabs independent.
  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey) ?? '{}');
    const count = value => Number.isSafeInteger(value) && value >= 0 ? value : 0;
    for (const model of config.models) {
      for (const source of sources) {
        const key = `${model.id}:${source}`;
        const row = saved?.[key];
        if (!row) continue;
        buckets[key] = {
          uncertain: count(row.uncertain), warming: count(row.warming),
          time: Number.isFinite(row.time) && row.time >= 0 ? row.time : 0,
          labels: Object.fromEntries(labelKeys.map(label => [label, count(row.labels?.[label])])),
        };
      }
    }
  } catch (_) {
    // Browsers that block site storage still support results in memory.
  }

  function save() {
    try { sessionStorage.setItem(storageKey, JSON.stringify(buckets)); } catch (_) { /* storage unavailable */ }
  }

  for (const [key, label] of Object.entries(config.labels)) {
    const row = document.createElement('li');
    const name = document.createElement('span');
    const track = document.createElement('span');
    const fill = document.createElement('span');
    const value = document.createElement('span');
    name.textContent = label;
    track.className = 'score-track';
    track.setAttribute('aria-hidden', 'true');
    fill.className = 'score-fill';
    value.className = 'score-number';
    track.append(fill);
    row.append(name, track, value);
    byId('stats-labels').append(row);
    labelRows.set(key, { fill, value });
  }

  for (const model of config.models) {
    const name = `${model.id.toUpperCase()} · ${model.name}`;
    modelFilter.add(new Option(name, model.id));
    const row = byId('stats-model-rows').insertRow();
    const heading = document.createElement('th');
    heading.scope = 'row';
    heading.textContent = name;
    row.append(heading);
    modelRows.set(model.id, { row, total: row.insertCell(), accepted: row.insertCell(), latency: row.insertCell() });
  }

  function aggregate(modelFilterValue, sourceFilterValue) {
    const result = empty(labelKeys);
    for (const [key, row] of Object.entries(buckets)) {
      const [model, source] = key.split(':');
      if (modelFilterValue !== 'all' && modelFilterValue !== model) continue;
      if (sourceFilterValue !== 'all' && sourceFilterValue !== source) continue;
      result.uncertain += row.uncertain;
      result.warming += row.warming;
      result.time += row.time;
      for (const label of labelKeys) result.labels[label] += row.labels[label];
    }
    return result;
  }

  function render() {
    const model = modelFilter.value;
    const source = sourceFilter.value;
    const tally = aggregate(model, source);
    const samples = total(tally);
    const labeled = accepted(tally);
    byId('stats-total').textContent = number.format(samples);
    byId('stats-accepted').textContent = number.format(labeled);
    byId('stats-rate').textContent = samples ? `${percent.format(labeled / samples)} tổng lượt` : '—';
    byId('stats-uncertain').textContent = number.format(tally.uncertain);
    byId('stats-warming').textContent = `${number.format(tally.warming)} đang lấy mẫu`;
    byId('stats-latency').textContent = samples ? `${decimal.format(tally.time / samples)} ms` : '—';
    for (const [label, { fill, value }] of labelRows) {
      value.textContent = number.format(tally.labels[label]);
      fill.style.width = `${labeled ? tally.labels[label] / labeled * 100 : 0}%`;
    }
    for (const [id, cells] of modelRows) {
      cells.row.hidden = model !== 'all' && id !== model;
      if (cells.row.hidden) continue;
      const row = aggregate(id, source);
      const size = total(row);
      cells.total.textContent = number.format(size);
      cells.accepted.textContent = size ? percent.format(accepted(row) / size) : '—';
      cells.latency.textContent = size ? decimal.format(row.time / size) : '—';
    }
    reset.disabled = Object.keys(buckets).length === 0;
  }

  modelFilter.addEventListener('change', render);
  sourceFilter.addEventListener('change', render);
  reset.addEventListener('click', () => {
    if (!window.confirm('Xóa toàn bộ thống kê trong tab này?')) return;
    buckets = {};
    save();
    render();
  });
  modelFilter.disabled = false;
  sourceFilter.disabled = false;
  render();

  return {
    record(model, source, label, warmingUp, latencyMs) {
      const key = `${model}:${source}`;
      const bucket = buckets[key] ??= empty(labelKeys);
      if (warmingUp) bucket.warming += 1;
      else if (label !== null) bucket.labels[label] += 1;
      else bucket.uncertain += 1;
      bucket.time += latencyMs;
      save();
      render();
    },
  };
}
