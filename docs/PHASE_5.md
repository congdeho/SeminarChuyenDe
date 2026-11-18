# Phase 5 · So sánh mô hình trên TrashNet

Ngày 18/11/2026, chủ đề được **thu hẹp theo yêu cầu**: bỏ phần chụp webcam trực tiếp. Vì vậy 60 ảnh `camera_dev`, 300 ảnh `camera_final`, thử webcam 10 phút và kết luận về domain shift TrashNet → ảnh tự chụp **không có dữ liệu**. Phase 4 chỉ có ứng dụng và video validation giả lập, không được gọi là demo webcam thật. Phase 5 vẫn so sánh mô hình trên TrashNet và chỉ mở test TrashNet sau khi khóa kiến trúc, checkpoint, ngưỡng và quy tắc chọn.

## Giao thức hiện dùng

- E1: CNN bốn block Conv–BatchNorm–ReLU–MaxPool, adaptive average pooling, dropout 0,3, sáu đầu ra; khởi tạo ngẫu nhiên, tối đa 40 epoch, augmentation có.
- E2: ResNet18 pretrained ImageNet, sáu đầu ra, cùng lịch huấn luyện Phase 3, augmentation có.
- E3: MobileNetV3-Small pretrained, augmentation có. `configs/phase5_e3.yaml` có cùng ánh xạ YAML với `configs/train.yaml`; checkpoint seed 42 cũ chỉ tái dùng khi checksum manifest/cấu hình khớp.
- E4: E3 nhưng không augmentation. Đây là ablation; ảnh validation vẫn dùng đúng tiền xử lý như E3.
- Seed 42/123/2026 như kế hoạch; mỗi run có thư mục riêng. Khi thiếu run, chỉ báo cáo số run đã hoàn tất; không lấy một seed đại diện cho ba seed.
- Dữ liệu train/validation dùng manifest khóa Phase 2. `src.evaluate` mặc định chỉ đánh giá validation. Test TrashNet yêu cầu tệp khóa JSON có `allow_trashnet_test`, checksum checkpoint/manifest và ngưỡng đã chọn.
- Ngưỡng quét 0,00–0,95 mỗi 0,05, lấy coverage cao nhất trong những điểm có accepted accuracy ≥0,90 và coverage ≥0,70; nếu không có thì dùng 0,65 và báo không đạt. Đây là ngưỡng **ảnh tĩnh**, không chứng minh phát hiện vật lạ.
- Benchmark batch 1, một ảnh validation lặp, warm-up 20 và đồng bộ MPS/CUDA; số `model_only_fps_estimate` không phải FPS webcam.

## Kết quả đã xác nhận

- E3 seed 42 từ Phase 3: validation 378 ảnh, accuracy 0,8730, macro-F1 0,8637. `runs/phase5/e3_seed42_val/predictions.csv` và `metrics.json` cho phép truy từng ảnh và ma trận nhầm lẫn. Không dùng test.
- E1 seed 42 đã huấn luyện **đủ 40 epoch** trên MPS, best epoch 40; evaluator trên 378 ảnh validation xác nhận accuracy 0,7222 và macro-F1 0,7031. Checkpoint `runs/phase5/e1_seed42/best.pt` SHA-256 `865c2d9f…`; `runs/phase5/e1_seed42_val/` lưu dự đoán và metrics. E1 thấp hơn E3 0,1606 macro-F1 ở seed 42; chưa suy rộng ra ba seed.
- E2 seed 42 ResNet18 pretrained: 8 epoch classifier + 9 fine-tune, dừng sớm; best fine-tune epoch 4 (global epoch 12). Evaluator trên 378 ảnh validation xác nhận accuracy 0,8783 và macro-F1 0,8681. Checkpoint `runs/phase5/e2_seed42/best.pt` SHA-256 `611f9b90…`; dự đoán và metrics trong `runs/phase5/e2_seed42_val/`. Chênh E2–E3 ở seed này chỉ +0,0044 macro-F1; chưa kết luận trước ba seed.
- E4 seed 42 MobileNet **không augmentation**: 8 classifier + 14 fine-tune epoch, dừng sớm; best fine-tune epoch 9 (global epoch 17). Evaluator xác nhận accuracy 0,8519 và macro-F1 0,8403; checkpoint `runs/phase5/e4_seed42/best.pt` SHA-256 `533cc4c8…`, dự đoán/metrics trong `runs/phase5/e4_seed42_val/`. E3 cao hơn E4 **0,0233 macro-F1** ở seed 42; đây là kết quả ablation một seed, chưa chứng minh lợi ích ổn định qua seed.
- Ứng viên ngưỡng E3 seed 42: **0,65**, 342/378 ảnh nhận (coverage 0,9048), accepted accuracy 0,9035; xem `runs/phase5/e3_seed42_val/threshold_sweep_v2.json` có checksum tệp dự đoán. E1 seed 42 không đạt đồng thời coverage ≥0,70 và accepted accuracy ≥0,90 ở các ngưỡng quét, nên `threshold_sweep.json` của E1 ghi fallback 0,65 và `target_met=false`. Chưa khóa ngưỡng mô hình cuối trước khi so sánh.
- E2 seed 42 có ứng viên ngưỡng **0,60** trên validation: 345/378 ảnh nhận (coverage 0,9127), accepted accuracy 0,9043; `runs/phase5/e2_seed42_val/threshold_sweep.json`. Không so trực tiếp “độ tự tin” giữa hai kiến trúc như thước đo xác suất đã hiệu chuẩn.
- E4 seed 42 có ứng viên ngưỡng **0,70** trên validation: 336/378 ảnh nhận (coverage 0,8889), accepted accuracy 0,9018; `runs/phase5/e4_seed42_val/threshold_sweep.json`. Cả ba ngưỡng E2/E3/E4 mới là **ứng viên theo từng checkpoint**.
- Benchmark E3 seed 42 CPU model-only **sơ bộ, chạy khi E1 đang huấn luyện MPS nên không dùng để chọn mô hình**: median 29,58 ms, p95 34,37 ms, 1.524.006 tham số; `runs/phase5/e3_seed42_benchmark_cpu.json`. Cần đo lại tất cả phương án khi máy rảnh. Không dùng số này như FPS toàn pipeline.
- Benchmark MPS model-only trên cùng Mac khi máy không huấn luyện: E1 median 1,93 ms/p95 2,89 ms, 390.918 tham số; E3 median 3,95 ms/p95 5,47 ms, 1.524.006 tham số. Nguồn `runs/phase5/e1_seed42_benchmark_mps.json` và `e3_seed42_benchmark_mps.json`. Đây là model forward batch 1 trên một ảnh validation lặp, **không phải FPS camera**; chưa dùng để chọn mô hình trước khi E2 và các seed xong.
- E2 model-only MPS trên cùng phép đo: median 3,15 ms/p95 4,11 ms, 11.179.590 tham số; `runs/phase5/e2_seed42_benchmark_mps.json`. E2 nhanh hơn E3 trong phép đo này dù nhiều tham số hơn; backend/hình dạng toán tử có thể ảnh hưởng latency. Kích thước `best.pt` là checkpoint **huấn luyện** có optimizer/scheduler, không phải kích thước xuất triển khai tối thiểu.
- E4 model-only MPS median 3,74 ms/p95 4,09 ms; `runs/phase5/e4_seed42_benchmark_mps.json`. Chênh E3/E4 latency nhỏ không có ý nghĩa so sánh kiến trúc vì cùng MobileNetV3-Small; số đo chịu nhiễu hệ thống.
- 23/23 kiểm thử kỹ thuật qua, gồm ba mô hình sáu đầu ra, đóng băng backbone, ablation deterministic, metrics biết trước đáp án, quy tắc chọn từ ba seed và hàng rào khóa test.
- Đã tạo `runs/phase5/e3_seed42_val/error_gallery_top12.png` từ **12 lỗi validation có điểm dự đoán sai cao nhất**, không phải 12 lỗi đại diện ngẫu nhiên. Nhóm quan sát: vật giấy có bề mặt sáng/trong hoặc hình khối đặc biệt (`paper370`, `paper521`, `paper10`, `paper107`) bị gán glass/plastic; giấy in hoặc hộp in (`paper314`, `paper53`, `cardboard190`) lẫn cardboard/paper; lon kim loại bóng/tối (`metal264`, `metal278`) bị gán glass; bao bì nhàu (`plastic461`, `metal353`) bị gán trash; chai thủy tinh mờ (`glass269`) bị gán plastic. Đây là **giả thuyết từ hình dạng/bề mặt**, không chứng minh nguyên nhân nội tại của mô hình. Điểm sai 0,888–0,999 nhắc rằng softmax cao không bảo đảm đúng và ngưỡng 0,65 không từ chối các ca này.

## Lệnh tái chạy

```bash
uv run --locked python -m src.train --config configs/phase5_e1.yaml --run-dir runs/phase5/e1_seed42
uv run --locked python -m src.train --config configs/phase5_e2.yaml --run-dir runs/phase5/e2_seed42
uv run --locked python -m src.train --config configs/phase5_e4.yaml --run-dir runs/phase5/e4_seed42
uv run --locked python -m src.evaluate --checkpoint runs/local_pretrained_probe/best.pt --split val --output-dir runs/phase5/e3_seed42_val
uv run --locked python -m src.select_threshold --predictions runs/phase5/e3_seed42_val/predictions.csv --output runs/phase5/e3_seed42_val/threshold_sweep.json
uv run --locked python -m src.benchmark --checkpoint runs/local_pretrained_probe/best.pt --device cpu --output runs/phase5/e3_seed42_benchmark_cpu.json
```

Mỗi lệnh tạo đầu ra mới và từ chối ghi đè; các đường dẫn ví dụ trên đã có kết quả ở máy này. Nếu một run dừng có chủ đích ở `--stop-after-epochs`, tiếp tục bằng `--resume` với đúng cấu hình. ResNet18 cần trọng số chính thức tải vào `.cache/torch/hub/checkpoints` (đã tải trên máy này); không gọi E2 pretrained khi chạy `--no-pretrained`.

Với hai seed còn lại, dùng cùng lệnh nhưng thêm `--seed 123` hoặc `--seed 2026`, đồng thời đặt `--run-dir` riêng theo mã và seed (ví dụ `runs/phase5/e2_seed123`). Khi `--resume`, phải truyền lại cùng `--seed`; checkpoint từ chối cấu hình khác. `--no-pretrained` chỉ được phép trong smoke run, không phải thí nghiệm chính.

## Kết quả đã khóa ngày 18/11/2026

Đủ 12 lượt chạy trên Mac MPS. Macro-F1 validation trung bình qua ba seed là 0,7074 cho E1, 0,8818 cho E2, 0,8791 cho E3 và 0,8625 cho E4. Augmentation tăng trung bình **1,66 điểm phần trăm macro-F1** khi đối chiếu E3/E4. E2 và E3 đủ điều kiện lựa chọn. E2 được chọn vì median latency 3,15 ms so với 3,95 ms của E3 trong phép đo forward batch 1 trên Mac MPS.

Tệp `runs/phase5/final/selection_lock.json` chốt ResNet18 seed 42, ngưỡng 0,60 và smoothing 7/10 trước khi đánh giá test. Trên 378 ảnh test, E2 đạt accuracy **88,89%**, macro-F1 **87,02%** và weighted-F1 **88,89%**. Ngưỡng 0,60 nhận 348 ảnh, trong đó 322 ảnh đúng. E3 đạt macro-F1 test 87,23%. Quyết định triển khai E2 giữ theo giao thức đã khóa bằng validation và latency.

Video offline với checkpoint đã khóa đọc và xuất đủ 120 frame. Bốn trong sáu đoạn kết thúc với nhãn ổn định đúng. Đoạn `paper` kết thúc ở trạng thái chưa chắc chắn và đoạn `plastic` kết thúc ở nhãn `glass`. Phần webcam thật và domain shift vẫn nằm ngoài phạm vi đã điều chỉnh. [Phân tích 12 ảnh sai trên test](PHAN_TICH_LOI_PHASE_5.md) · [Bảng kết quả đầy đủ](../outputs/phase6/ban_ket_qua/KET_QUA.md).
