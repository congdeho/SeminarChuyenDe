# Seminar phân loại chất thải rắn

Ứng dụng web cục bộ dùng PyTorch để phân loại một vật chính trong ảnh hoặc khung hình. Ứng dụng nhận ảnh tải lên, video và webcam. Sáu nhãn gồm **bìa carton, thủy tinh, kim loại, giấy, nhựa và rác khác**. Giao diện hiển thị điểm theo lớp, trạng thái dự đoán và thống kê theo mô hình, nguồn đầu vào.

**E2 · ResNet18** là mô hình mặc định. Bộ chọn còn có **E1 · CNN cơ sở**, **E3 · MobileNetV3-Small có augmentation** và **E4 · MobileNetV3-Small không augmentation**. E2 có thể tự gọi **E5** để phân biệt một chai nhựa với một chai thủy tinh khi hai vật liệu cùng có điểm cao.

## Cài đặt và chạy web

Các lệnh dưới đây dùng shell trên macOS, Linux hoặc WSL. Cần Git, GitHub CLI, `uv`, kết nối mạng để cài lần đầu và quyền đọc repository GitHub. `uv` quản lý Python 3.12 và cài thư viện theo `uv.lock`. [Cài uv](https://docs.astral.sh/uv/getting-started/installation/) · [Cài GitHub CLI](https://cli.github.com/).

### 1. Clone và cài thư viện

```bash
gh auth login
gh repo clone congdeho/SeminarChuyenDe
cd SeminarChuyenDe
uv python install 3.12
uv sync --locked
uv run --locked --offline python -m src.check_env
```

Bỏ qua `gh auth login` nếu GitHub CLI đã đăng nhập. Nếu đang ở thư mục dự án có sẵn, chạy từ bước cài Python.

### 2. Tải checkpoint đã huấn luyện

Gói khoảng **209 MiB** chứa checkpoint E1–E5 và tệp khóa E5. Giải nén từ thư mục gốc dự án để các tệp vào đúng vị trí dưới `runs/`. [Gói checkpoint v0.1.0](https://github.com/congdeho/SeminarChuyenDe/releases/tag/v0.1.0).

```bash
mkdir -p artifacts
gh release download v0.1.0 --repo congdeho/SeminarChuyenDe --pattern seminar-checkpoints.zip --dir artifacts
printf '%s  %s\n' '1c58eb218d7e29ab7b48ceb9cd20f7089bfb4a306d2df4aa3805e59af63f6876' 'artifacts/seminar-checkpoints.zip' | shasum -a 256 -c - && unzip -nq artifacts/seminar-checkpoints.zip
```

Tiếp tục giải nén khi checksum trả về `OK`. Nếu máy đã có checkpoint, có thể dùng các tệp hiện có. E2 cần `runs/phase5/e2_seed42/best.pt`.

Để bật E5 tự động, tải thêm trọng số bộ phát hiện chai của Torchvision một lần vào cache của dự án. Nếu bỏ qua bước này, E2 vẫn phân loại sáu lớp.

```bash
XDG_CACHE_HOME="$PWD/.cache" uv run --locked --offline python -c 'from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as W; W.COCO_V1.get_state_dict(progress=True, check_hash=True, weights_only=True)'
```

Lệnh trên dùng gói Python đã cài và tải trọng số từ PyTorch nếu cache còn thiếu. [Faster R-CNN MobileNetV3 của Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html).

### 3. Mở ứng dụng

```bash
uv run --locked --offline python -m src.web_app
```

Mở [http://127.0.0.1:8765](http://127.0.0.1:8765). Ứng dụng mặc định chạy trên CPU và lắng nghe ở máy cục bộ. Dùng `--port 8766` nếu cổng bận. Có thể thêm `--device mps` trên Mac có MPS. Dừng máy chủ bằng `Ctrl+C`.

- **Ảnh** · Chọn một ảnh JPEG, PNG hoặc WebP tối đa 20 MiB. Ứng dụng phân loại ngay. Đổi mô hình để thử lại trên cùng ảnh.
- **Video** · Chọn tệp tối đa 150 MiB rồi bấm **Phân loại video**. Ứng dụng lấy mẫu theo thời gian và làm mượt dự đoán. Đổi mô hình rồi chạy lại để so sánh.
- **Webcam** · Bấm **Bật webcam**, cấp quyền Camera trong trình duyệt và đặt một vật vào ô giữa. Kết quả ổn định cần đủ 10 khung hình, ít nhất 7 phiếu cùng nhãn và điểm trung bình đạt ngưỡng mô hình. Bấm **Tắt webcam** khi xong.
- **Thống kê** · Lọc theo mô hình hoặc nguồn. Một lượt là một ảnh hoặc một khung hình đã xử lý. Số liệu tổng hợp được giữ trong cùng tab và có thể đặt lại bằng **Xóa thống kê**.

Bạn có thể dùng ảnh, video của mình hoặc webcam ngay sau khi có checkpoint. Hai nút **Dùng ảnh mẫu** và **Dùng video mẫu** dùng dữ liệu chuẩn bị ở mục tiếp theo.

## Dữ liệu mẫu và tái huấn luyện

[TrashNet của Gary Thung và Mindy Yang](https://github.com/garythung/trashnet) cung cấp dữ liệu phân loại sáu lớp. Manifest còn **2.521 ảnh** sau khi loại sáu tệp trùng byte nhưng mâu thuẫn nhãn. Tập train có 1.765 ảnh, validation có 378 ảnh và test có 378 ảnh. [Kiểm tra và chia dữ liệu](docs/DU_LIEU_PHASE_2.md).

```bash
mkdir -p data/raw
curl -fL -o data/raw/trashnet-dataset-resized.zip https://raw.githubusercontent.com/garythung/trashnet/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip
printf '%s  %s\n' '0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2' 'data/raw/trashnet-dataset-resized.zip' | shasum -a 256 -c - && unzip -nq data/raw/trashnet-dataset-resized.zip 'dataset-resized/*' -d data/raw
uv run --locked --offline python -m src.prepare_data
uv run --locked --offline python -m src.check_data
uv run --locked --offline python -m src.make_camera_fixture
```

`src.make_camera_fixture` tạo video từ ảnh validation và yêu cầu đường dẫn đầu ra chưa tồn tại. Nếu đã có `runs/camera_fixture/validation_sequence.mp4`, bỏ qua lệnh cuối để dùng video hiện có. Với dữ liệu này, có thể thử đúng ca chai nhựa dễ nhầm bằng tệp `data/raw/dataset-resized/plastic/plastic171.jpg`.

Để huấn luyện lại E2 từ manifest trên và thử suy luận ảnh từ Terminal, dùng thư mục run mới nếu đường dẫn dưới đây đã có checkpoint. Kết quả huấn luyện có thể khác theo phần cứng và môi trường.

```bash
uv run --locked python -m src.train --config configs/phase5_e2.yaml --run-dir runs/phase5/e2_seed42
uv run --locked --offline python -m src.predict --checkpoint runs/phase5/e2_seed42/best.pt --image data/raw/dataset-resized/plastic/plastic171.jpg
```

Lệnh `src.predict` trả điểm của checkpoint truyền vào. Luồng tự gọi E5 được dùng trong ứng dụng web. Xem thêm [cấu hình và giao thức thí nghiệm](docs/PHASE_5.md) và [notebook huấn luyện Colab](notebooks/01_huan_luyen_colab.ipynb).

## Kết quả và phạm vi thực nghiệm

- **E2 seed 42** đạt **accuracy 88,89%** và **macro-F1 87,02%** trên 378 ảnh test TrashNet sau khi khóa lựa chọn dựa trên validation. Bốn cấu hình được chạy với ba seed 42, 123 và 2026, tổng cộng 12 lượt. [Kết quả Phase 5](docs/PHASE_5.md).
- **E5** dùng dữ liệu vật liệu từ TrashNet và vùng chai đã gán nhãn của [TACO](https://github.com/pedropro/TACO). Trên 58 vùng chai test TACO, E5 đúng 49/58, so với 44/58 của E2 giới hạn cùng hai nhãn. Phép đo này dùng vùng chai cắt sẵn. Pipeline E2 tự gọi E5 qua bộ phát hiện chai là phần thử nghiệm riêng.
- Ứng dụng hỗ trợ webcam. Phần đánh giá đã hoàn tất dùng ảnh TrashNet và video dựng từ ảnh validation. Chất lượng trên webcam thật vẫn cần đo trên bộ ảnh tự chụp.

### Khi nào E2 gọi E5

Nhựa và thủy tinh phải là hai lớp dẫn đầu, có tổng điểm từ 0,90 và lớp đứng thứ hai đạt ít nhất 0,05. Bộ phát hiện cần tìm một chai có điểm từ 0,30 và mọi chai bổ sung dưới 0,20. Khi E5 đạt 0,65, ứng dụng chia lại tổng điểm nhựa và thủy tinh theo tỷ lệ E5, giữ điểm bốn lớp còn lại. Giao diện hiển thị điểm E2 ban đầu và ý kiến E5. Lượt này thuộc thống kê E2. Các ngưỡng gọi là quy tắc thử nghiệm. [Mã gọi E5](src/bottle_routing.py).

## Mã nguồn và kiểm thử

```text
src/          Chuẩn bị dữ liệu, huấn luyện, đánh giá và suy luận
src/web_ui/   Giao diện ảnh, video, webcam và thống kê
configs/      Cấu hình huấn luyện cho từng thí nghiệm
notebooks/    Kiểm tra môi trường và huấn luyện trên Colab
tests/        Kiểm thử dữ liệu, mô hình, camera và điều kiện gọi E5
docs/         Báo cáo, quy trình tái lập và tài liệu bảo vệ
```

```bash
uv run --locked --offline python -m unittest discover -s tests -v
```

Git theo dõi mã nguồn, cấu hình và tài liệu. Dữ liệu, cache, checkpoint và kết quả chạy được lưu riêng trong `data/`, `.cache/`, `artifacts/`, `runs/` và `outputs/`.

[Báo cáo seminar](docs/BAO_CAO_SEMINAR.md) · [Gói demo và báo cáo Phase 6](docs/PHASE_6.md) · [Kịch bản bảo vệ](docs/BAO_VE_SEMINAR.md).
