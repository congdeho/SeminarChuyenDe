# Kiểm thử

Phase 2: từ thư mục dự án, sau khi chuẩn bị TrashNet và manifest, chạy `uv run --locked python -m unittest discover -s tests -v`. Kiểm tra nhãn, cả sáu lớp trong ba tập, rò rỉ SHA/group, tái lập manifest, tensor 3×224×224, val xác định và train augmentation thay đổi. `src.check_data` kiểm tra batch train/val và tạo hình ví dụ biến đổi; không nạp test.

Phase 3: kiểm tra đầu ra sáu lớp, đóng băng backbone/BatchNorm, dự đoán không đổi sau lưu/nạp checkpoint và (nếu có smoke artifact) history sau resume. Các kiểm thử này không tải pretrained weights hay mở tập test. Run thật và số liệu validation nằm trong `runs/local_pretrained_probe/`.

Phase 4: kiểm tra ROI, BGR→RGB/suy luận ảnh lưu không đổi, 10-frame smoothing không giữ nhãn cũ, cấu trúc metadata cho ảnh phát triển, và xử lý video MP4/headless với checkpoint thật. Video dùng ảnh validation, không dùng test hoặc webcam. Quyền Camera và 60 ảnh phát triển cần nghiệm thu riêng trên máy thật.

Phase 5: kiểm tra CNN/ResNet18/MobileNetV3 sáu đầu ra, đóng băng từng backbone, E3 tương thích cấu hình cũ, E4 chỉ thay augmentation, evaluator bằng đáp án biết trước, ngưỡng chỉ dùng validation, chọn mô hình từ ba seed và yêu cầu tệp khóa/checksum trước khi mở test. Tổng hiện tại 23/23 bài kiểm thử; các smoke run E2/E4 không phải kết quả báo cáo.

Phase 1 có công cụ kiểm tra môi trường `python -m src.check_env`. Phần chụp webcam trực tiếp đã tạm bỏ theo điều chỉnh phạm vi ngày 18/11/2026.
