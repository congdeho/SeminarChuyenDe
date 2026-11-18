import { createStatistics } from './stats.js';

const byId = id => document.getElementById(id);
const ui = Object.fromEntries([
  'connection', 'stage', 'empty', 'empty-title', 'empty-help', 'media', 'preview', 'camera', 'roi',
  'file-input', 'choose', 'sample', 'run', 'stop', 'source-note', 'video-time', 'video-progress',
  'model', 'model-note', 'state', 'prediction', 'prediction-detail', 'routing', 'score-type', 'scores', 'timing',
].map(id => [id, byId(id)]));

const modes = {
  image: { title: 'Chọn một ảnh để phân loại', help: 'JPEG, PNG hoặc WebP · tối đa 20 MiB',
    note: 'Chọn ảnh hoặc dùng ảnh mẫu để thử mô hình.', accept: 'image/jpeg,image/png,image/webp,image/bmp,image/gif',
    choose: 'Chọn ảnh', sample: 'Dùng ảnh mẫu', limit: 20 },
  video: { title: 'Chọn video để xem kết quả theo khung hình', help: 'MP4, MOV, AVI hoặc WebM · tối đa 150 MiB',
    note: 'Video được xem theo khung hình và không phát âm thanh.',
    accept: 'video/*,.mp4,.mov,.m4v,.avi,.webm,.mkv,.mpeg,.mpg', choose: 'Chọn video', sample: 'Dùng video mẫu', limit: 150 },
  webcam: { title: 'Bật webcam để xem trực tiếp', help: 'Đặt một vật vào vùng vuông ở giữa khung hình',
    note: 'Trình duyệt sẽ hỏi quyền truy cập camera.', choose: 'Bật webcam' },
};

const selected = { image: null, video: null };
const canvas = document.createElement('canvas');
const frameContext = canvas.getContext('2d');
const percent = new Intl.NumberFormat('vi-VN', { style: 'percent', maximumFractionDigits: 1 });
let mode = 'image';
let config = null;
let models = new Map();
let statistics = null;
let job = null;
let imageUrl = null;
let dimensions = null;
let cameraStream = null;
let cameraRequest = null;
let cameraTimer = null;
let cameraStartId = 0;
let cameraStarting = false;
let windowScores = [];
let lastVideoTime = -1;
const scoreRows = new Map();

function note(text, error = false) {
  if (ui['source-note'].textContent !== text) ui['source-note'].textContent = text;
  ui['source-note'].dataset.error = String(error);
}

function setState(text, state = '') {
  ui.state.textContent = text;
  ui.state.dataset.state = state;
}

function resetResult() {
  setState('Chờ dữ liệu');
  ui.prediction.textContent = 'Chưa phân loại';
  ui['prediction-detail'].textContent = 'Kết quả sẽ hiển thị ở đây.';
  ui.routing.hidden = true;
  ui['score-type'].textContent = 'Softmax';
  ui.timing.textContent = '—';
  for (const row of scoreRows.values()) {
    row.element.dataset.winner = 'false';
    row.fill.style.width = '0%';
    row.number.textContent = '—';
  }
}

function showPrediction(result, stable = null) {
  const option = models.get(result.model);
  const label = key => config.labels[key] ?? key;
  const routing = result.routing;
  const scoreType = routing?.applied ? 'điểm kết hợp' : 'điểm softmax';
  const warmingUp = stable?.status === 'warming_up';
  const isAccepted = stable ? stable.status === 'stable' : result.confidence >= option.threshold;
  const winner = isAccepted ? (stable?.label ?? result.label) : null;
  if (warmingUp) {
    setState('Đang làm mượt');
    ui.prediction.textContent = 'Đang lấy mẫu';
    ui['prediction-detail'].textContent = `${stable.count}/${config.window} khung hình · Gợi ý ${label(result.label)}`;
  } else if (isAccepted) {
    setState(stable ? 'Ổn định' : 'Đạt ngưỡng', 'stable');
    ui.prediction.textContent = label(winner);
    ui['prediction-detail'].textContent = stable
      ? `${percent.format(stable.mean_score)} điểm trung bình · ${stable.votes}/${config.window} phiếu`
      : `${percent.format(result.confidence)} ${scoreType} · ${winner}`;
  } else {
    setState('Chưa chắc chắn', 'uncertain');
    ui.prediction.textContent = 'Chưa chắc chắn';
    const source = stable ? 'Khung hiện tại' : 'Gợi ý';
    ui['prediction-detail'].textContent = `${source} · ${label(result.label)} · ${percent.format(result.confidence)}`;
  }
  ui.routing.hidden = !routing;
  if (routing) {
    const base = `${label(routing.base_label)} ${percent.format(routing.base_confidence)}`;
    const specialist = `${label(routing.specialist_label)} ${percent.format(routing.specialist_confidence)}`;
    ui.routing.textContent = routing.applied
      ? `E2 ban đầu · ${base} → E5 · ${specialist}`
      : `E5 · ${specialist}, dưới ngưỡng. Giữ kết quả E2.`;
  }
  ui['score-type'].textContent = routing?.applied ? 'Kết hợp E2 và E5' : 'Softmax';
  for (const [key, row] of scoreRows) {
    const score = result.scores[key];
    row.element.dataset.winner = String(key === result.label);
    row.fill.style.width = `${score * 100}%`;
    row.number.textContent = percent.format(score);
  }
  ui.timing.textContent = `${result.model.toUpperCase()} · ${result.timing_ms.model_ms.toFixed(1)} ms suy luận · ${config.device.toUpperCase()}`;
  statistics.record(result.model, mode, winner, warmingUp, result.timing_ms.model_ms);
}

function scoringWindow(result) {
  windowScores.push(result);
  if (windowScores.length > config.window) windowScores.shift();
  if (windowScores.length < config.window) return { status: 'warming_up', count: windowScores.length };
  const votes = new Map();
  for (const frame of windowScores) votes.set(frame.label, (votes.get(frame.label) ?? 0) + 1);
  const [label, count] = [...votes].sort((a, b) => b[1] - a[1])[0];
  const mean = windowScores.reduce((sum, frame) => sum + frame.scores[label], 0) / config.window;
  const stable = count >= config.min_votes && mean >= models.get(result.model).threshold;
  return { status: stable ? 'stable' : 'uncertain', label, votes: count, mean_score: mean };
}

function fitMedia() {
  if (!dimensions) return;
  const { width, height, bounds } = dimensions;
  const scale = Math.min(ui.stage.clientWidth / width, ui.stage.clientHeight / height);
  ui.media.style.width = `${width * scale}px`;
  ui.media.style.height = `${height * scale}px`;
  if (bounds) {
    const [x, y, w, h] = bounds;
    [ui.roi.style.left, ui.roi.style.top, ui.roi.style.width, ui.roi.style.height] =
      [x, y, w, h].map(value => `${value * scale}px`);
  }
}

function showMedia(element, width, height, bounds = null) {
  if (!width || !height) return;
  dimensions = { width, height, bounds };
  ui.empty.hidden = true;
  ui.media.hidden = false;
  ui.preview.hidden = element !== ui.preview;
  ui.camera.hidden = element !== ui.camera;
  ui.roi.hidden = !bounds;
  fitMedia();
}

function clearMedia() {
  dimensions = null;
  ui.media.hidden = true;
  ui.empty.hidden = false;
  ui.preview.hidden = true;
  ui.preview.removeAttribute('src');
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null;
  ui.roi.hidden = true;
}

function controls() {
  const valid = Boolean(models.get(ui.model.value)?.available);
  const isVideo = mode === 'video';
  const isCamera = mode === 'webcam';
  const stopping = isCamera ? cameraStream || cameraStarting : isVideo && job;
  ui.choose.textContent = modes[mode].choose;
  ui.choose.hidden = Boolean(isCamera && cameraStream);
  ui.choose.disabled = !valid || cameraStarting;
  ui.sample.hidden = isCamera;
  ui.sample.disabled = !valid;
  ui.sample.textContent = modes[mode].sample ?? '';
  ui.run.hidden = !isVideo || !selected.video;
  ui.run.disabled = !valid || Boolean(job);
  ui.stop.hidden = !stopping;
  ui.stop.textContent = isCamera ? 'Tắt webcam' : 'Dừng phân loại';
}

function cancelJob() {
  job?.abort();
  job = null;
}

function finishJob(controller) {
  if (job !== controller) return;
  job = null;
  controls();
}

function resetCameraScoring() {
  clearTimeout(cameraTimer);
  cameraRequest?.abort();
  cameraRequest = null;
  windowScores = [];
  lastVideoTime = -1;
}

function stopCamera() {
  cameraStartId += 1;
  cameraStarting = false;
  resetCameraScoring();
  const oldStream = cameraStream;
  cameraStream = null;
  oldStream?.getTracks().forEach(track => track.stop());
  ui.camera.pause();
  ui.camera.srcObject = null;
  ui.camera.hidden = true;
}

function selectMode(next) {
  if (next === mode) return;
  cancelJob();
  stopCamera();
  clearMedia();
  resetResult();
  mode = next;
  for (const button of document.querySelectorAll('[data-mode]')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
  }
  ui['empty-title'].textContent = modes[mode].title;
  ui['empty-help'].textContent = modes[mode].help;
  ui['video-time'].hidden = true;
  ui['video-progress'].hidden = true;
  note(mode === 'video' && selected.video ? selected.video.name : modes[mode].note);
  controls();
  if (mode === 'image' && selected.image) runImage(selected.image, true);
}

function modelDescription() {
  const option = models.get(ui.model.value);
  ui['model-note'].textContent = option
    ? `${option.description} Ngưỡng ${option.threshold.toLocaleString('vi-VN', { minimumFractionDigits: 2 })}.`
    : 'Chưa có checkpoint để chạy.';
  for (const [key, row] of scoreRows) row.element.hidden = option?.labels ? !option.labels.includes(key) : false;
}

async function apiResponse(url, settings = {}) {
  const response = await fetch(url, settings);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ?? `Máy chủ trả mã ${response.status}.`);
  }
  return response;
}

function post(url, body, controller) {
  return apiResponse(url, { method: 'POST', body, signal: controller.signal });
}

function showError(error) {
  if (error.name !== 'AbortError') {
    setState('Có lỗi', 'uncertain');
    note(error.message || 'Không thực hiện được yêu cầu.', true);
  }
}

function chooseFile(file, sourceMode, sample = false) {
  if (!file || sourceMode !== mode) return;
  if (!file.size || file.size > modes[mode].limit * 1024 * 1024) {
    note(`Chọn tệp có dữ liệu và nhỏ hơn ${modes[mode].limit} MiB.`, true);
    return;
  }
  cancelJob();
  clearMedia();
  resetResult();
  ui['video-time'].hidden = true;
  ui['video-progress'].hidden = true;
  selected[mode] = file;
  note(file.name);
  controls();
  if (mode === 'image') runImage(file, true);
  if (mode === 'video' && sample) runVideo(file);
}

async function runImage(file, preview) {
  cancelJob();
  resetResult();
  if (!models.get(ui.model.value)?.available) return;
  if (preview) {
    clearMedia();
    const url = URL.createObjectURL(file);
    imageUrl = url;
    ui.preview.onload = () => {
      if (mode === 'image' && imageUrl === url) showMedia(ui.preview, ui.preview.naturalWidth, ui.preview.naturalHeight);
    };
    ui.preview.src = url;
  }
  const controller = new AbortController();
  job = controller;
  setState('Đang phân loại');
  note(file.name);
  try {
    const response = await post(`/api/predict?model=${ui.model.value}`, file, controller);
    const result = await response.json();
    if (job === controller) showPrediction(result);
  } catch (error) {
    if (job === controller) showError(error);
  } finally {
    finishJob(controller);
  }
}

function timeText(value) {
  const seconds = Math.floor(value);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

async function runVideo(file) {
  cancelJob();
  clearMedia();
  resetResult();
  const controller = new AbortController();
  job = controller;
  controls();
  setState('Đang mở video');
  note(file.name);
  ui['video-progress'].hidden = true;
  ui['video-time'].hidden = true;
  const extension = file.name.split('.').pop().toLowerCase();
  let duration = 0;
  let count = 0;
  let finished = false;
  function handle(line) {
    if (!line || job !== controller) return;
    const packet = JSON.parse(line);
    if (packet.type === 'error') throw new Error(packet.error);
    if (packet.type === 'video') duration = packet.duration_s;
    if (packet.type === 'done') finished = true;
    if (packet.type !== 'frame') return;
    count += 1;
    ui.preview.src = `data:image/jpeg;base64,${packet.jpeg}`;
    showMedia(ui.preview, packet.frame_width, packet.frame_height, packet.roi);
    showPrediction(packet, packet.stable);
    ui['video-time'].textContent = duration ? `${timeText(packet.time_s)} / ${timeText(duration)}` : timeText(packet.time_s);
    ui['video-time'].hidden = false;
    if (duration) {
      ui['video-progress'].value = Math.min(1, packet.time_s / duration);
      ui['video-progress'].hidden = false;
    }
  }
  try {
    const url = `/api/video?model=${ui.model.value}&format=${encodeURIComponent(extension)}`;
    const response = await post(url, file, controller);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const chunk = await reader.read();
      buffer += decoder.decode(chunk.value, { stream: !chunk.done });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      lines.forEach(handle);
      if (chunk.done) break;
    }
    handle(buffer);
    if (job === controller) {
      if (!finished) throw new Error('Kết nối kết thúc trước khi xử lý xong video.');
      note(`${file.name} · Đã phân loại ${count} khung hình.`);
      if (duration) {
        ui['video-progress'].value = 1;
        ui['video-time'].textContent = `${timeText(duration)} / ${timeText(duration)}`;
      }
    }
  } catch (error) {
    if (job === controller) {
      showError(error);
      controller.abort();
    }
  } finally {
    finishJob(controller);
  }
}

function cameraLoop(delay = 120) {
  clearTimeout(cameraTimer);
  if (cameraStream && mode === 'webcam') cameraTimer = setTimeout(pollCamera, delay);
}

async function pollCamera() {
  if (!cameraStream || mode !== 'webcam' || cameraRequest) return;
  if (ui.camera.readyState < 2 || lastVideoTime === ui.camera.currentTime) {
    cameraLoop();
    return;
  }
  lastVideoTime = ui.camera.currentTime;
  const controller = new AbortController();
  cameraRequest = controller;
  const scale = Math.min(1, 720 / ui.camera.videoWidth);
  canvas.width = Math.round(ui.camera.videoWidth * scale);
  canvas.height = Math.round(ui.camera.videoHeight * scale);
  try {
    frameContext.drawImage(ui.camera, 0, 0, canvas.width, canvas.height);
    const frame = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.86));
    if (cameraRequest !== controller) return;
    if (!frame) throw new Error('Không thể đọc khung hình webcam.');
    const response = await post(`/api/predict?source=webcam&model=${ui.model.value}`, frame, controller);
    const result = await response.json();
    if (cameraRequest !== controller) return;
    showMedia(ui.camera, canvas.width, canvas.height, result.roi);
    showPrediction(result, scoringWindow(result));
    note('Webcam đang bật · Đặt một vật vào vùng nhận dạng.');
  } catch (error) {
    if (cameraRequest === controller) showError(error);
  } finally {
    if (cameraRequest === controller) {
      cameraRequest = null;
      cameraLoop();
    }
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    note('Webcam cần trình duyệt hỗ trợ Camera và địa chỉ localhost.', true);
    return;
  }
  const startId = ++cameraStartId;
  cameraStarting = true;
  controls();
  resetResult();
  note('Đang xin quyền truy cập webcam…');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 720 } }, audio: false });
    if (startId !== cameraStartId || mode !== 'webcam') {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    cameraStream = stream;
    for (const track of stream.getVideoTracks()) track.addEventListener('ended', () => {
      if (cameraStream !== stream) return;
      stopCamera();
      clearMedia();
      controls();
      note('Webcam đã ngắt kết nối.', true);
    });
    ui.camera.srcObject = stream;
    await ui.camera.play();
    if (cameraStream !== stream) return;
    showMedia(ui.camera, ui.camera.videoWidth, ui.camera.videoHeight);
    cameraLoop(0);
  } catch (error) {
    if (startId !== cameraStartId) return;
    stopCamera();
    note(error.name === 'NotAllowedError' ? 'Chưa có quyền Camera. Kiểm tra quyền của trang trong trình duyệt.'
      : error.name === 'NotFoundError' ? 'Không tìm thấy webcam.'
      : 'Không mở được webcam. Kiểm tra camera và các ứng dụng đang sử dụng.', true);
  } finally {
    if (startId === cameraStartId) cameraStarting = false;
    controls();
  }
}

document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => selectMode(button.dataset.mode)));
ui.model.addEventListener('change', () => {
  modelDescription();
  cancelJob();
  resetResult();
  if (mode === 'image' && selected.image) runImage(selected.image, false);
  if (mode === 'video') {
    clearMedia();
    ui['video-time'].hidden = true;
    ui['video-progress'].hidden = true;
    note(selected.video ? `${selected.video.name} · Nhấn Phân loại video để chạy với mô hình mới.` : modes.video.note);
  }
  if (mode === 'webcam') {
    resetCameraScoring();
    cameraLoop(0);
  }
  controls();
});
ui.choose.addEventListener('click', () => {
  if (mode === 'webcam') return startCamera();
  ui['file-input'].accept = modes[mode].accept;
  ui['file-input'].value = '';
  ui['file-input'].click();
});
ui['file-input'].addEventListener('change', () => chooseFile(ui['file-input'].files[0], mode));
ui.sample.addEventListener('click', async () => {
  const sourceMode = mode;
  const fileName = sourceMode === 'image' ? 'sample.jpg' : 'sample.mp4';
  cancelJob();
  const controller = new AbortController();
  job = controller;
  controls();
  try {
    const response = await apiResponse(`/${fileName}`, { signal: controller.signal });
    const blob = await response.blob();
    if (job === controller) chooseFile(new File([blob], fileName, { type: blob.type }), sourceMode, true);
  } catch (error) {
    if (job === controller) showError(error);
  } finally {
    finishJob(controller);
  }
});
ui.run.addEventListener('click', () => { if (selected.video) runVideo(selected.video); });
ui.stop.addEventListener('click', () => {
  if (mode === 'webcam') {
    stopCamera();
    clearMedia();
    resetResult();
    note('Webcam đã tắt.');
  } else if (mode === 'video') {
    cancelJob();
    note('Đã dừng phân loại video.');
    if (!dimensions) resetResult();
  }
  controls();
});
window.addEventListener('pagehide', () => { stopCamera(); cancelJob(); });
new ResizeObserver(fitMedia).observe(ui.stage);

async function init() {
  try {
    config = await (await apiResponse('/api/models')).json();
    models = new Map(config.models.map(option => [option.id, option]));
    statistics = createStatistics(config);
    ui.model.replaceChildren(...config.models.map(option => {
      const text = `${option.id.toUpperCase()} · ${option.name}${option.available ? '' : ' (thiếu checkpoint)'}`;
      const item = new Option(text, option.id);
      item.disabled = !option.available;
      return item;
    }));
    const active = models.get(config.default)?.available ? config.default : config.models.find(option => option.available)?.id;
    if (active) ui.model.value = active;
    ui.model.disabled = !active;
    for (const [key, label] of Object.entries(config.labels)) {
      const element = document.createElement('li');
      const name = document.createElement('span');
      const score = document.createElement('span');
      const fill = document.createElement('span');
      const number = document.createElement('span');
      name.textContent = label;
      score.className = 'score-track';
      fill.className = 'score-fill';
      number.className = 'score-number';
      score.append(fill);
      element.append(name, score, number);
      ui.scores.append(element);
      scoreRows.set(key, { element, fill, number });
    }
    ui.connection.dataset.state = 'ready';
    ui.connection.textContent = `Cục bộ · ${config.device.toUpperCase()}`;
    modelDescription();
    resetResult();
    controls();
  } catch (error) {
    ui.connection.dataset.state = 'error';
    ui.connection.textContent = 'Mất kết nối';
    showError(error);
  }
}

init();
