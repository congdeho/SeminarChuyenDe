# Phase 6 · Đóng gói và luyện bảo vệ

## Bản kết quả Phase 6

Mở `outputs/phase6/ban_ket_qua/`. Báo cáo có kết quả ba seed, điểm test của bốn checkpoint seed 42, chỉ số từng lớp, ma trận nhầm lẫn và phân tích 12 ảnh. PowerPoint có tám slide với số liệu đã khóa. ZIP đi kèm chứa ResNet18 seed 42, video 120 frame, ảnh mẫu, mã nguồn, cấu hình và checksum. [Báo cáo](../outputs/phase6/ban_ket_qua/BAO_CAO_SEMINAR.md) · [PowerPoint](../outputs/phase6/ban_ket_qua/Seminar_TrashNet.pptx) · [Gói demo](../outputs/phase6/ban_ket_qua/Demo_Seminar_TrashNet.zip).

## Bản luyện bảo vệ trước khi chạy test

Thư mục `outputs/phase6/ban_luyen_bao_ve/` lưu bản luyện cũ với MobileNetV3-Small và kết quả validation seed 42. Dùng `ban_ket_qua/` để trình chiếu kết quả cuối.

Chế độ demo CPU đã chạy từ bản giải nén với một môi trường mới cài bằng `uv sync --locked --offline`. Ảnh `sample.jpg` trả nhãn `cardboard`. Video xuất đủ 120 frame và dừng tại cuối tệp. Trong phạm vi đã điều chỉnh, dữ liệu demo là chuỗi lặp từ sáu ảnh TrashNet validation. Điều kiện webcam thật chưa có số đo.

Trong bản luyện, bốn trong sáu đoạn kết thúc với nhãn ổn định đúng. Đoạn `paper` kết thúc ở `plastic` và đoạn `plastic` kết thúc ở `glass`. Có thể dùng hai đoạn này khi giải thích giới hạn của ROI và smoothing. Các frame là ảnh lặp, nên tỷ lệ frame đúng trên chuỗi không đại diện cho độ chính xác trên nhiều vật thể.

## Trình chiếu và tái lập

Phát `demo.mp4` ngay sau slide demo. Để suy luận lại khi chuẩn bị báo cáo, giải nén gói và chạy từ thư mục vừa giải nén.

```bash
shasum -a 256 -c SHA256SUMS
uv sync --locked
uv run --locked --offline python -m src.predict --checkpoint best.pt --image sample.jpg --device cpu
uv run --locked --offline python -m src.camera --checkpoint best.pt --input-video validation_sequence.mp4 --headless --record --device cpu --threshold 0.60 --run-dir repro_demo
```

Lệnh trên dùng ngưỡng 0,60 của ResNet18 seed 42. Với máy đã có bộ cài trong cache, chạy `uv sync --locked --offline`. Nếu gặp lỗi thiếu dependency, hoàn tất bước cài trước buổi trình bày. Nếu thư mục đầu ra đã chứa kết quả, dùng một `--run-dir` mới.

## Nguồn số liệu

`python -m src.finalize_trashnet` đối chiếu đầy đủ cấu hình và checksum, chọn checkpoint từ validation, ghi tệp khóa, kiểm tra video offline rồi đánh giá bốn checkpoint seed 42 trên 378 ảnh test. `python -m src.package_seminar --output outputs/phase6/ban_ket_qua` đóng gói báo cáo và demo theo checkpoint đã khóa. Số liệu và bằng chứng chi tiết nằm ở `runs/phase5/final/` và `evidence/` trong ZIP bản kết quả.

Lịch sử từng lượt huấn luyện nằm trong các tệp `history.csv` theo seed. Nhật ký khâu đánh giá và đóng gói nằm ở `outputs/phase6/finalize.log`.

Buổi tập trình bày và thông tin cá nhân trên trang bìa do người trình bày hoàn tất. Dùng [bài nói ba phút và 12 câu hỏi](BAO_VE_SEMINAR.md) để luyện trước khi trình bày.
