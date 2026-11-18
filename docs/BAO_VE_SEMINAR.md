# Luyện bảo vệ seminar

## Bài nói ba phút

**0:00–0:35 · Nêu bài toán.** Mỗi ảnh hoặc ROI có một vật thể chính. Mục tiêu là phân loại sáu nhãn TrashNet. Em triển khai cả chế độ ảnh và luồng frame OpenCV. Phần demo dự phòng được dựng từ ảnh validation.

**0:35–1:10 · Giải thích dữ liệu.** Em kiểm tra ảnh trùng, loại sáu tệp mâu thuẫn nhãn, gom các ứng viên gần trùng và khóa manifest với 1.765 ảnh train, 378 validation và 378 test. Em chọn macro-F1 vì muốn mỗi lớp có trọng số bằng nhau trong chỉ số tổng hợp.

**1:10–2:00 · Mô hình và kết quả.** Chỉ vào E1–E4 trên slide. E1 là baseline từ đầu. E2/E3 tái sử dụng trọng số ImageNet rồi thay classifier sáu lớp và fine-tune. E3/E4 khác ở augmentation. Nêu rằng ResNet18 có macro-F1 validation trung bình 88,18% qua ba seed và median latency 3,15 ms. Checkpoint seed 42 đạt accuracy 88,89% và macro-F1 87,02% trên test TrashNet.

**2:00–2:40 · Demo.** Phát MP4 cục bộ. Chỉ vùng vuông ở giữa, nhãn thô và nhãn sau smoothing. Giải thích vì sao cần đủ 10 frame và ít nhất 7 phiếu. Nêu một chuyển cảnh sai hoặc chưa ổn định để cho thấy giới hạn.

**2:40–3:00 · Kết luận.** Nêu phương án có bằng chứng tốt nhất trong phiên bản báo cáo. Nhấn phạm vi đo gồm TrashNet và luồng video mô phỏng. Việc kiểm chứng trên webcam và ảnh tự chụp còn thiếu dữ liệu.

## Câu hỏi phản biện

1. **Vì sao classification?** Mỗi ROI giả định một vật thể chính và mô hình dự đoán nhãn cho cả ROI. Detection cần nhãn vị trí và có thể xử lý nhiều vật. Bằng chứng nằm ở `src/camera.py` và classifier sáu đầu ra trong `src/model.py`.
2. **Trash khác uncertain thế nào?** Trash là một lớp học từ dữ liệu. Uncertain là trạng thái do luật ứng dụng trả về từ điểm và số phiếu. Vật ngoài sáu lớp có thể vẫn nhận một nhãn với điểm cao.
3. **Chống rò rỉ thế nào?** So SHA để phát hiện trùng byte, gom ứng viên gần trùng trước khi chia, lưu manifest và checksum. Dữ liệu thiếu ID vật thể đầy đủ nên có thể sót ảnh cùng vật.
4. **Vì sao đóng băng backbone?** Trọng số ImageNet cung cấp đặc trưng ban đầu. Giai đoạn classifier thích ứng đầu dự đoán. Đặt BatchNorm của backbone ở eval giúp giữ thống kê đã học trong giai đoạn đóng băng. Fine-tune cho phép backbone thích ứng với TrashNet.
5. **Accuracy khác macro-F1 thế nào?** Accuracy đếm số ảnh đúng trên tổng số ảnh. Macro-F1 cho mỗi lớp một phần bằng nhau, kể cả lớp ít ảnh. Xem precision, recall và support theo lớp trong `metrics.json`.
6. **Có thể nói kiến trúc ResNet tự nó tốt hơn CNN không?** Kết quả gộp cả tác động của kiến trúc, khởi tạo và chế độ huấn luyện. So sánh E1/E2 là so sánh phương án. E3/E4 là cặp kiểm tra augmentation có kiểm soát hơn.
7. **Augmentation có tăng bao nhiêu?** E3 đạt macro-F1 validation trung bình 87,91% qua ba seed, E4 đạt 86,25%, chênh 1,66 điểm phần trăm. Seed 2026 cho kết quả gần như ngang nhau, nên hiệu ứng quan sát được thay đổi theo seed.
8. **Ngưỡng từ chối thay đổi kết quả thế nào?** ResNet18 với ngưỡng 0,60 nhận 348/378 ảnh test (92,06%). Trong số được nhận, 322 ảnh đúng (92,53%), tương ứng 85,19% đúng và được nhận trên toàn test. Ngưỡng được chọn bằng validation trước test.
9. **Một ảnh sai tự tin cao?** `plastic469.jpg` trên test của E2 seed 42 bị gán `glass` với điểm 99,87%. Thành chai xanh trong và cong tạo hình ảnh giống vật thủy tinh. Đây là một giả thuyết từ quan sát ảnh.
10. **Latency và FPS webcam khác nhau thế nào?** Benchmark forward batch 1 đồng bộ MPS sau warm-up. Chạy webcam còn thêm thời gian nhận frame, ROI, tiền xử lý và hiển thị. Một frame đổi nhãn của video dựng chỉ đo phản ứng trên chuỗi đó.
11. **Ảnh camera thật giảm bao nhiêu?** Chưa có dữ liệu ảnh tự chụp nên chưa định lượng độ suy giảm. Góc, ánh sáng, nền và chất liệu là các điều kiện cần thử khi thu dữ liệu thật.
12. **Thí nghiệm tiếp theo là gì?** Tách vật thể theo ID trước khi chụp train/dev/test. Chọn ngưỡng trên dev rồi đo trên vật thể chưa từng dùng. Quay thêm cảnh trống và vật ngoài lớp nếu muốn đánh giá tỷ lệ báo nhầm.

## Phương án trình chiếu

Mở slide và video MP4 trước buổi báo cáo. Video lấy từ bộ ảnh validation. Dùng một ảnh có trong gói để chứng minh suy luận lại bằng CPU khi môi trường đã cài. Thông tin người trình bày và thời lượng chính thức của môn cần bổ sung trước khi nộp. Buổi tập nói và trả lời phản biện do người trình bày trực tiếp thực hiện.
