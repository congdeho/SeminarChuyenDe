# Phân loại chất thải rắn từ ảnh

Seminar chuyên đề · Báo cáo phương pháp và thực nghiệm TrashNet

## Tóm tắt

Đề tài xây dựng bộ phân loại một vật thể theo sáu nhãn TrashNet. Dự án triển khai CNN cơ sở, ResNet18 và MobileNetV3-Small, đồng thời khảo sát ảnh hưởng của augmentation trên MobileNetV3-Small. Ứng dụng OpenCV nhận ảnh từ webcam hoặc tệp video, cắt vùng quan tâm ở giữa khung hình và làm mượt dự đoán qua nhiều frame. Phạm vi thực nghiệm hiện tại dùng ảnh TrashNet và video dựng từ ảnh validation. Bộ ảnh tự chụp và nghiệm thu webcam trực tiếp được lược bỏ theo phạm vi đã điều chỉnh.

## 1. Bài toán

Mỗi ảnh hoặc vùng quan tâm chứa tối đa một vật thể chính. Mô hình trả về một trong sáu nhãn `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`. `trash` là lớp có mẫu gán nhãn riêng trong TrashNet. Trạng thái `uncertain` do ứng dụng tạo khi điểm thấp hoặc dự đoán thiếu ổn định. Một khung hình có nhiều vật thể thuộc bài toán phát hiện hoặc phân đoạn đối tượng, nằm ngoài phạm vi thực nghiệm này.

## 2. Dữ liệu và chia tập

TrashNet có 2.527 ảnh. Dự án loại sáu tệp thuộc ba cặp trùng byte nhưng mâu thuẫn nhãn và giữ 2.521 ảnh trong manifest. Các cặp gần trùng cùng nhãn do kiểm tra pHash và dHash phát hiện được gom nhóm trước khi chia tập. Chia gần theo tỷ lệ 70/15/15 cho 1.765 ảnh train, 378 ảnh validation và 378 ảnh test. Nhóm ảnh gần trùng là biện pháp giảm rò rỉ. Việc nhận diện cùng một vật thể từ nhiều góc vẫn có sai số vì TrashNet thiếu ID vật thể nhất quán. Lớp `trash` có 20 mẫu test, nên chỉ số lớp này nhạy với từng ảnh.

## 3. Phương pháp

- **E1** dùng CNN bốn khối tích chập với BatchNorm, ReLU và MaxPool. Mô hình khởi tạo ngẫu nhiên, sử dụng pooling thích nghi, dropout và đầu ra sáu lớp.
- **E2** dùng ResNet18, khởi tạo backbone bằng trọng số ImageNet của Torchvision. Thay đầu phân loại bằng lớp tuyến tính sáu đầu ra.
- **E3** dùng MobileNetV3-Small với trọng số ImageNet của Torchvision và augmentation.
- **E4** giữ cấu hình E3 và thay phép biến đổi train bằng phép tiền xử lý xác định. So sánh E3/E4 đo ảnh hưởng của cả nhóm augmentation dùng trong dự án.

Pipeline đánh giá đổi ảnh sang RGB, resize cạnh ngắn lên 256, crop giữa 224 × 224 và chuẩn hóa theo ImageNet. Với augmentation, train dùng random crop, lật ngang, xoay và thay đổi màu nhẹ. Huấn luyện dùng cross-entropy với trọng số lớp tính riêng từ train. Hai phương án pretrained huấn luyện classifier trước với backbone đóng băng, sau đó fine-tune toàn mạng với learning rate thấp hơn. Checkpoint tốt nhất lấy theo macro-F1 validation. F1 của từng lớp là trung bình điều hòa giữa precision và recall. Macro-F1 lấy trung bình đều sáu lớp, còn weighted-F1 dùng số mẫu từng lớp làm trọng số.

## 4. Giao thức đánh giá

E1–E4 dùng cùng manifest và ba seed 42, 123, 2026. Tiêu chí chọn demo dùng trung bình macro-F1 validation của E1–E3. Trong nhóm cách giá trị tốt nhất tối đa 0,02, chọn latency trung vị thấp nhất trên Mac MPS với batch size 1. Nếu hòa, chọn ít tham số hơn. Checkpoint seed 42 của phương án được chọn dùng cho demo. Trước khi đọc test, dự án khóa checksum checkpoint, manifest và ngưỡng ảnh tĩnh lấy từ validation. Ngưỡng tối ưu theo quy tắc đã định nếu coverage đạt ít nhất 70% và accuracy trên ảnh chấp nhận đạt ít nhất 90%.

## 5. Kết quả thực nghiệm

Đã hoàn thành 12 lượt huấn luyện. ResNet18 seed 42 được khóa cho demo. Trên 378 ảnh test TrashNet, checkpoint đạt accuracy 88,89%, macro-F1 87,02% và weighted-F1 88,89%. Với ngưỡng từ chối 0,60, checkpoint nhận 348/378 ảnh (92,06%), trong đó 322 ảnh đúng (92,53%).

Trên validation, augmentation tăng macro-F1 trung bình của MobileNet 1,66 điểm phần trăm qua ba seed. Kết quả E1 và E2 phản ánh cả kiến trúc lẫn cách khởi tạo. Báo cáo bản kết quả có bảng từng seed, chỉ số từng lớp, confusion matrix và kết quả benchmark. [Báo cáo kèm số liệu](../outputs/phase6/ban_ket_qua/BAO_CAO_SEMINAR.md).

Trên test, ResNet18 gán `plastic469.jpg` thành `glass` với softmax 99,87%. Cả 12 lỗi có điểm cao nhất đều vượt ngưỡng 0,60. [Quan sát 12 ảnh sai](PHAN_TICH_LOI_PHASE_5.md).

## 6. Triển khai và giới hạn

OpenCV đọc frame BGR. Ứng dụng lấy ROI vuông ở giữa, đổi sang RGB, dùng tiền xử lý của checkpoint rồi tính sáu điểm softmax. Smoothing yêu cầu đủ 10 dự đoán, nhãn thắng ít nhất 7 phiếu và điểm trung bình của nhãn đó đạt ngưỡng. Smoothing giảm chớp nhãn và có thể khiến nhãn đổi chậm hoặc ổn định ở một dự đoán sai. Video dựng từ ảnh validation kiểm tra đọc frame, ROI, overlay, trạng thái chuyển tiếp và ghi tệp. Benchmark forward MPS đo riêng mô hình đã warm-up. Chất lượng, tốc độ webcam và mức suy giảm khi đổi sang ảnh tự chụp vẫn cần dữ liệu thực tế.

## 7. Phần triển khai và phần tái sử dụng

Dự án xây dựng manifest, quy trình chia tập, CNN cơ sở, vòng huấn luyện và resume, bộ đánh giá, phép đo latency và ứng dụng camera. Dữ liệu TrashNet, kiến trúc ResNet18, MobileNetV3-Small và trọng số ImageNet do các nguồn bên dưới cung cấp. Quá trình chuẩn bị mã và tài liệu có sử dụng Codex. Khi nộp bài, ghi rõ mức hỗ trợ theo quy định môn học và điền thông tin học viên lên trang bìa nếu giảng viên yêu cầu.

## Tài liệu tham khảo

1. Yang, M. và Thung, G. *TrashNet*. [Kho dữ liệu và mã nguồn](https://github.com/garythung/trashnet).
2. He, K., Zhang, X., Ren, S. và Sun, J. *Deep Residual Learning for Image Recognition*. [Bản công bố](https://arxiv.org/abs/1512.03385).
3. Howard, A. và cộng sự. *Searching for MobileNetV3*. [Bản công bố](https://arxiv.org/abs/1905.02244).
4. Torchvision. [Models and pre-trained weights](https://docs.pytorch.org/vision/stable/models.html).
