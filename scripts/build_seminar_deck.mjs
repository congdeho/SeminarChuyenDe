import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const output = path.resolve(process.argv[2]);
const skill = process.env.SEMINAR_PRESENTATION_SKILL;
process.env.RUNTIME_NODE_MODULES = process.env.SEMINAR_NODE_MODULES;
const require = createRequire(path.join(process.env.SEMINAR_NODE_MODULES, 'package.json'));
const { Presentation, PresentationFile } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')));
const { finalizePresentation } = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const data = JSON.parse(await fs.readFile(path.join(output, 'slide_data.json'), 'utf8'));
const isFinal = Object.keys(data.test).length > 0;
const stage = await fs.mkdtemp(path.join(path.dirname(output), `.slides-build-${path.basename(output)}-`));
console.log(`Slide renders: ${stage}`);

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const palette = { paper: '#F6F5ED', ink: '#102B2A', accent: '#176B5B', muted: '#475D5B', white: '#FFFFFF' };
const font = 'Arial';
const percent = value => value === null ? 'N/A' : new Intl.NumberFormat('vi-VN', {
  style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2,
}).format(value);
const number = value => new Intl.NumberFormat('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const trashnet = 'https://github.com/garythung/trashnet';

function txt(slide, text, left, top, width, height, settings = {}) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left, top, width, height }, fill: 'none' });
  shape.text = text;
  shape.text.style = { typeface: font, fontSize: '21pt', color: palette.ink, verticalAlignment: 'top',
    insets: { left: 0, right: 0, top: 0, bottom: 0 }, ...settings };
  return shape;
}

function slide(title, notes) {
  const page = deck.slides.add();
  page.background.fill = palette.paper;
  if (title) txt(page, title, 72, 58, 1136, 85, { fontSize: '36pt', bold: true });
  page.speakerNotes.text = notes;
  return page;
}

async function picture(page, file, left, top, width, height, alt) {
  const bytes = await fs.readFile(path.resolve(root, file));
  page.images.add({ blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: file.endsWith('.png') ? 'image/png' : 'image/jpeg', fit: 'contain',
    position: { left, top, width, height }, alt });
}

function list(page, lines, start = 170, left = 72, width = 1100) {
  lines.forEach(([label, body], index) => {
    const top = start + 133 * index;
    txt(page, label, left, top, width, 39, { fontSize: '24pt', bold: true, color: palette.accent });
    txt(page, body, left, top + 45, width, 70);
  });
}

function table(page, values, widths, top = 160) {
  const rows = values.length;
  const columns = values[0].length;
  const grid = page.tables.add({ rows, columns, left: 72, top, width: 1136, height: rows * 62,
    columnWidths: widths, values });
  for (let row = 0; row < rows; row++) {
    grid.rows[row].height = 62;
    for (let column = 0; column < columns; column++) {
      const cell = grid.getCell(row, column);
      cell.fill = row === 0 ? palette.ink : row % 2 ? palette.white : '#E9F0E8';
      cell.text.style = { typeface: font, fontSize: '17pt', bold: row === 0,
        color: row === 0 ? palette.white : palette.ink };
    }
  }
  grid.borders.assign({ fill: '#D4DFD8', style: 'solid', width: 1 });
}

const cover = slide(null, `Seminar chuyên đề. Ảnh minh họa là glass105.jpg từ TrashNet validation. Nguồn ${trashnet}`);
cover.background.fill = palette.ink;
txt(cover, 'Phân loại\nchất thải rắn từ ảnh', 72, 178, 650, 245,
  { fontSize: '50pt', bold: true, color: palette.paper });
txt(cover, 'Seminar chuyên đề · TrashNet', 76, 460, 650, 60, { fontSize: '22pt', color: '#BDE4CF' });
await picture(cover, 'data/raw/dataset-resized/glass/glass105.jpg', 755, 110, 455, 490, 'Ảnh thủy tinh từ TrashNet');

const problem = slide('Bài toán phân loại một vật thể', `Nguồn nhãn và ảnh ${trashnet}. Trình bày giả định một vật thể chính nằm trong ROI.`);
list(problem, [['Đầu vào', 'Ảnh tĩnh hoặc vùng vuông giữa khung hình video.'],
  ['Sáu nhãn TrashNet', 'Bìa carton, thủy tinh, kim loại, giấy, nhựa và trash.'],
  ['Trạng thái chưa chắc chắn', 'Ứng dụng hoãn trả nhãn khi điểm thấp hoặc dự đoán thiếu ổn định.']], 170, 72, 675);
await picture(problem, 'data/raw/dataset-resized/plastic/plastic1.jpg', 770, 154, 430, 410, 'Ảnh nhựa từ TrashNet');

const dataset = slide('Dữ liệu và chia tập', `TrashNet ${trashnet}. Manifest SHA-256 eea2338ba07941cd61abff4703734126b0275f4a51bad4d664d09ab4f63dd1de.`);
[['1.765', 'Train'], ['378', 'Validation'], ['378', 'Test']].forEach(([count, label], index) => {
  const left = 72 + 383 * index;
  txt(dataset, count, left, 180, 320, 110, { bold: true, fontSize: '62pt', color: palette.accent });
  txt(dataset, label, left + 3, 298, 320, 60, { fontSize: '25pt' });
});
txt(dataset, 'Loại sáu ảnh trùng byte có nhãn mâu thuẫn. Gom các ảnh gần trùng trước khi chia tập.', 72, 415, 1100, 80);
txt(dataset, 'Giới hạn của manifest: TrashNet thiếu mã định danh vật thể đầy đủ.', 72, 535, 1100, 70,
  { color: palette.muted });

const method = slide('Phương án huấn luyện', 'ResNet: https://arxiv.org/abs/1512.03385\nMobileNetV3: https://arxiv.org/abs/1905.02244\nTrọng số: https://docs.pytorch.org/vision/stable/models.html');
list(method, [['E1 · CNN cơ sở', 'Bốn khối tích chập. Khởi tạo ngẫu nhiên.'],
  ['E2 và E3 · Transfer learning', 'ResNet18 và MobileNetV3-Small dùng trọng số ImageNet. Huấn luyện classifier sáu lớp trước khi fine-tune.'],
  ['E3 và E4 · Augmentation', 'Hai cấu hình MobileNet dùng cùng kiến trúc và lịch huấn luyện. E4 tắt augmentation.']]);

const validation = slide('So sánh trên 378 ảnh validation', 'Số liệu từ runs/phase5. Latency là median forward MPS batch 1, warm-up 20 lượt và 100 lượt đo.');
const names = { e1: 'E1  CNN cơ sở', e2: 'E2  ResNet18', e3: 'E3  MobileNet + aug.', e4: 'E4  MobileNet' };
if (isFinal) {
  table(validation, [['Phương án', 'Macro-F1 TB', 'SD (điểm %)', 'Median MPS'],
    ...Object.entries(data.validation).map(([key, row]) => [names[key], percent(row.mean_macro_f1),
      number(row.sd_macro_f1 * 100), `${number(row.median_ms)} ms`])], [324, 268, 270, 274]);
} else {
  table(validation, [['Phương án', 'Accuracy seed 42', 'Macro-F1 seed 42', 'Median MPS'],
    ...Object.entries(data.validation).map(([key, row]) => [names[key], percent(row.accuracy),
      percent(row.macro_f1), `${number(row.median_ms)} ms`])], [324, 268, 270, 274]);
}
const pilotDelta = data.validation.e3.macro_f1 - data.validation.e4.macro_f1;
const delta = isFinal ? data.validation.e3.mean_macro_f1 - data.validation.e4.mean_macro_f1 : pilotDelta;
txt(validation, `E3 − E4: ${number(delta * 100)} điểm phần trăm macro-F1 ${isFinal ? 'trung bình ba seed' : 'ở seed 42'}.`,
  72, 507, 1136, 80, { color: palette.accent, bold: true });
if (isFinal) {
  const eligible = data.eligible.map(key => key.toUpperCase()).join(', ');
  txt(validation, `Chọn ${data.selected.toUpperCase()} theo latency trong nhóm ${eligible} cách điểm cao nhất tối đa 0,02.`,
    72, 565, 1136, 65);
}

const evalSlide = slide(isFinal ? 'Đánh giá trên 378 ảnh test TrashNet' : 'Ngưỡng từ chối trên validation',
  isFinal ? 'Accuracy và F1 trong bảng dùng argmax sáu lớp trên toàn bộ 378 ảnh. Dòng cuối áp dụng ngưỡng đã khóa trên validation cho checkpoint demo.'
    : 'Giải thích coverage và accuracy trên các ảnh được nhận. Bản luyện lấy ngưỡng của E3 seed 42.');
if (isFinal) {
  table(evalSlide, [['Phương án seed 42','Accuracy','Macro-F1','Weighted-F1'],
    ...Object.entries(data.test).map(([key, row]) => [names[key], percent(row.accuracy), percent(row.macro_f1), percent(row.weighted_f1)])],
    [350, 245, 245, 296]);
  const row = data.test[data.selected];
  txt(evalSlide, `Demo ${data.selected.toUpperCase()} · Ngưỡng ${number(data.threshold)} · Coverage ${percent(row.coverage)} · Accuracy ảnh nhận ${percent(row.accepted_accuracy)}`,
    72, 510, 1136, 100, { color: palette.accent, bold: true });
} else {
  list(evalSlide, [['0,65 · Ngưỡng ứng viên MobileNet', 'Chọn bằng dự đoán validation của E3 seed 42.'],
    ['90,48% · Coverage', 'Ứng dụng nhận 342 trong 378 ảnh validation.'],
    ['90,35% · Accuracy trên số ảnh nhận', 'Kết quả test còn chờ đủ ba seed và tệp khóa đánh giá.']]);
}

const error = isFinal ? data.errors[0] : null;
const errorPath = error?.path ?? 'data/raw/dataset-resized/paper/paper370.jpg';
const errorDesc = error ? `Ảnh ${error.true_label} bên phải bị gán ${error.predicted_label} với softmax ${percent(Number(error.confidence))}.`
  : 'MobileNet E3 gán ảnh giấy bên phải thành glass với softmax khoảng 0,999.';
const demo = slide('Demo offline và lỗi tự tin cao', `Ảnh ${path.basename(errorPath)} từ TrashNet ${isFinal ? 'test' : 'validation'}. Nguồn ${trashnet}. Mở demo.mp4 trong gói khi trình bày. Video dựng từ ảnh validation.`);
list(demo, [['Video 120 frame', 'Sáu ảnh validation lặp thành chuỗi. Ứng dụng cắt ROI, đổi RGB và chuẩn hóa.'],
  ['Smoothing', 'Chờ đủ 10 dự đoán. Nhãn cần ít nhất 7 phiếu và điểm trung bình đạt ngưỡng.'],
  [isFinal ? 'Một lỗi trên tập test' : 'Một lỗi validation', errorDesc]], 148, 72, 620);
await picture(demo, errorPath, 755, 177, 455, 325, errorDesc);

const conclusion = slide('Kết luận và phạm vi kiểm chứng', 'Nêu rõ phần triển khai của dự án và phần tái sử dụng. Chuẩn bị nêu một lỗi và trả lời cách chia tập.');
const chosen = isFinal
  ? `${data.selected.toUpperCase()} seed 42 đạt macro-F1 ${percent(data.test[data.selected].macro_f1)} trên test TrashNet.`
  : 'Kết quả hiện có là đối chiếu bốn cấu hình trên validation seed 42.';
list(conclusion, [['Kết quả thực nghiệm', chosen], ['Demo lặp lại được', 'Gói kèm checkpoint, ảnh đầu vào và video. Chế độ CPU có thể chạy sau khi cài dependency.'],
  ['Giới hạn', 'Chất lượng nhận dạng và tốc độ trên webcam thật còn cần được đo.']]);

for (const [index, page] of deck.slides.items.entries()) {
  const png = await deck.export({ slide: page, format: 'png', scale: 1 });
  await fs.writeFile(path.join(stage, `slide-${index + 1}.png`), Buffer.from(await png.arrayBuffer()));
}
const candidatePath = path.join(stage, 'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({ workspaceDir: root, candidatePath,
  finalPath: path.join(output, 'Seminar_TrashNet.pptx'), pythonExecutable: process.env.SEMINAR_RUNTIME_PYTHON,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--require-native-table-slide', '5',
    ...(isFinal ? ['--require-native-table-slide', '6'] : [])],
  requiredNativeTableOwnerSlides: isFinal ? [5, 6] : [5], explicitTotalSlideCount: 8,
  fontPolicy: { basis: 'design', families: [font] }, verifyArtifactToolImport: true,
  receiptPath: path.join(stage, 'validation.json') });
