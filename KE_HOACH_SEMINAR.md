# KẾ HOẠCH SEMINAR: PHÂN LOẠI CHẤT THẢI RẮN TỪ CAMERA

> **Cập nhật 06/10/2026:** Dự án chuyển sang một người thực hiện. Kế hoạch đang áp dụng là [Kế hoạch solo — 6 phase](KE_HOACH_SOLO_6_PHASE.md). Nội dung bên dưới được giữ làm tài liệu tham khảo kỹ thuật; phân công bốn người và trình tự cũ không còn áp dụng. Khi có khác biệt, ưu tiên kế hoạch solo.

**Quy mô:** Nhóm 4 người, mỗi người phụ trách một phần.  
**Thời gian:** 8 tuần kể từ ngày nhóm bắt đầu.  
**Công nghệ:** Python, PyTorch, OpenCV; huấn luyện trên Google Colab GPU, demo trên Mac.  
**Ngày lập:** 01/10/2026.  
**Trạng thái:** Kế hoạch triển khai; các chỉ số mục tiêu chưa phải kết quả thực nghiệm.

## 1. Mục tiêu và phạm vi

### 1.1. Tên đề tài

**Xây dựng và đánh giá hệ thống phân loại chất thải rắn từ camera sử dụng học chuyển giao.**

Đề tài cần hoàn thành cả huấn luyện mô hình và sử dụng mô hình với camera. Nhóm được tái sử dụng kiến trúc, trọng số pretrained và thư viện, nhưng phải ghi rõ nguồn, phần tự triển khai và kết quả thực nghiệm của nhóm.

### 1.2. Bài toán

- Phân loại ảnh đa lớp, đơn nhãn; mỗi ảnh hoặc vùng camera chứa một vật thể rác chính.
- Phân loại theo sáu nhãn của TrashNet.
- Hiển thị nhãn dự đoán, điểm softmax và trạng thái chưa chắc chắn.
- Đánh giá chất lượng phân loại và tốc độ vận hành thực tế.

| Nhãn trong mã nguồn | Hiển thị tiếng Việt |
|---|---|
| `cardboard` | Bìa carton |
| `glass` | Thủy tinh |
| `metal` | Kim loại |
| `paper` | Giấy |
| `plastic` | Nhựa |
| `trash` | Rác khác theo nhãn TrashNet |

**Phân biệt:** `trash` là một lớp có trong dữ liệu huấn luyện. “Không xác định” hoặc “Chưa chắc chắn” là trạng thái từ chối trả nhãn, không phải lớp thứ bảy đã được huấn luyện.

### 1.3. Phần bắt buộc

1. Dữ liệu sạch, cách chia tập cố định và kiểm tra rò rỉ dữ liệu.
2. CNN cơ sở, ResNet18 và MobileNetV3-Small.
3. So sánh mô hình và thí nghiệm tác động của augmentation.
4. Chương trình suy luận ảnh và webcam chạy offline.
5. Đánh giá trên test TrashNet và 300 ảnh tự chụp độc lập.
6. Báo cáo, slide, mã nguồn, cấu hình, checkpoint, hướng dẫn chạy và video dự phòng.

### 1.4. Ngoài phạm vi bắt buộc

YOLO, nhận dạng nhiều vật thể, phân đoạn ảnh, Raspberry Pi, ứng dụng web và cơ cấu phân loại rác tự động được đưa vào hướng phát triển. Hệ thống không kết luận chắc chắn khả năng tái chế chỉ từ nhãn vật liệu.

### 1.5. Mục tiêu kỹ thuật

- Mục tiêu ban đầu trên test TrashNet: accuracy từ 85% và macro-F1 từ 0,80.
- Mục tiêu demo: tối thiểu 15 FPS trên máy trình diễn, chạy liên tục 10 phút không lỗi.
- Hoàn thành đầy đủ đánh giá dữ liệu thực tế, kể cả khi kết quả thấp hơn TrashNet.
- Lưu đủ thông tin để tái lập thí nghiệm và truy nguồn mọi bảng kết quả.

Các ngưỡng là mục tiêu để hướng tới, không phải cam kết về kết quả hay điểm số. Nếu chưa đạt, báo cáo trung thực và phân tích nguyên nhân; không thay đổi test để làm đẹp kết quả.

## 2. Phân công và trách nhiệm

Điền tên thành viên trước khi bắt đầu. Mỗi phần có một người chịu trách nhiệm chính và một người kiểm tra bàn giao.

| Vai trò | Thành viên | Công việc chính | Người kiểm tra |
|---|---|---|---|
| Người 1 — Dữ liệu | … | Làm sạch, chia tập, quy tắc nhãn, ảnh thực tế | Người 4 |
| Người 2 — Mô hình | … | Pipeline huấn luyện, ba mô hình, checkpoint, log | Người 3 kiểm tra tích hợp; Người 4 kiểm tra thí nghiệm |
| Người 3 — Ứng dụng | … | Suy luận ảnh, webcam, smoothing, tốc độ | Người 2 và Người 4 |
| Người 4 — Đánh giá | … | Đánh giá độc lập, phân tích, điều phối và tổng hợp báo cáo | Cả nhóm |

Mỗi người viết phần chuyên môn của mình. Người 4 biên tập và thống nhất báo cáo, không chịu trách nhiệm viết thay toàn bộ nhóm. Cả bốn phải hiểu pipeline tổng thể.

## 3. Phần Người 1: Dữ liệu

### 3.1. Công việc theo thứ tự

1. Tải TrashNet từ nguồn tác giả, ghi đường dẫn và phiên bản dữ liệu sử dụng.
2. Kiểm tra ảnh đọc được, ảnh trùng chính xác, ảnh gần trùng, nhãn bất thường và số lượng từng lớp.
3. Tạo biểu đồ phân bố lớp và ảnh minh họa từng lớp.
4. Lập nhóm các ảnh xác định được là cùng vật thể hoặc gần trùng.
5. Chia train/validation/test xấp xỉ 70/15/15, seed `42`, cố gắng giữ tỷ lệ lớp và không tách một nhóm vật thể sang nhiều tập.
6. Xuất manifest cố định, báo cáo số lượng thực tế và mã kiểm tra dữ liệu.
7. Viết quy tắc chụp và gán nhãn dữ liệu camera.
8. Thu thập bộ phát triển camera và bộ kiểm thử cuối; kiểm tra chéo nhãn cùng Người 4.

### 3.2. Nguồn dữ liệu

- [TrashNet](https://github.com/garythung/trashnet): bộ dữ liệu chính; README tác giả mô tả 2.527 ảnh thuộc sáu lớp, phần lớn chụp vật thể trên nền sạch.
- [Waste datasets review](https://github.com/AgaMiko/waste-datasets-review): danh mục phục vụ khảo sát, không phải một bộ ảnh duy nhất.
- [TACO](https://github.com/pedropro/TACO): dùng trong tổng quan và hướng phát triển phát hiện/phân đoạn rác ngoài thực tế.
- [Thư mục Drive do giảng viên gợi ý](https://drive.google.com/drive/folders/1INjJjfTzF-MS6bevTgchNHJiWzFwk6Gq): chưa xác minh nội dung; không phụ thuộc vào nguồn này để hoàn thành kế hoạch.

Thống kê tham khảo của TrashNet: glass 501, paper 594, cardboard 403, plastic 482, metal 410, trash 137. Khi tải về phải đếm lại, ghi rõ mọi chênh lệch và mọi mẫu bị loại.

### 3.3. Quy tắc chống rò rỉ dữ liệu

- Chia tập trước khi augmentation; chỉ áp dụng augmentation trên train.
- Ảnh cùng vật thể hoặc gần trùng đã xác định phải nằm trong cùng tập.
- Chấp nhận tỷ lệ chia lệch nhẹ để giữ nhóm; công bố số mẫu và phân bố cuối cùng.
- Không khẳng định loại bỏ hoàn toàn trùng vật thể nếu thiếu mã định danh trong nguồn gốc.
- Không dùng test để chọn tham số, ngưỡng confidence, epoch hay checkpoint.
- Khóa phiên bản manifest; khi sửa dữ liệu phải ghi thay đổi và xác định thí nghiệm cần chạy lại.

Manifest tối thiểu:

```text
path,label,split,object_id,source
```

`path` là đường dẫn tương đối so với thư mục dữ liệu. `object_id` bắt buộc đối với ảnh tự chụp; với TrashNet chỉ ghi khi có căn cứ xác định. Lưu checksum manifest cùng kết quả thí nghiệm.

### 3.4. Ảnh tự chụp

| Bộ ảnh | Quy mô | Mục đích |
|---|---:|---|
| Phát triển camera | 60 ảnh, 10 ảnh/lớp | Kiểm tra tích hợp và hướng dẫn đặt vật |
| Kiểm thử cuối | 300 ảnh, 50 ảnh/lớp | Đánh giá domain shift sau khi khóa hệ thống |

- Hai bộ dùng vật thể khác nhau và không dùng để huấn luyện trong phạm vi này.
- Với bộ kiểm thử, mỗi lớp dùng khoảng 10 vật thể, mỗi vật thể chụp 5 góc hoặc điều kiện; tổng cộng 60 vật thể, 300 ảnh.
- Chụp qua ít nhất 3 buổi, có nhiều nền, ánh sáng và khoảng cách nhưng vẫn bảo đảm một vật thể chính nhìn thấy rõ.
- Báo cáo cả số ảnh và số vật thể vì nhiều ảnh cùng vật thể không độc lập hoàn toàn.
- Nhãn khó hoặc vật nhiều chất liệu được ghi vào danh sách ngoài phạm vi, không ép nhãn tùy ý.
- Người 1 và Người 4 kiểm tra chéo nhãn trước khi khóa bộ kiểm thử.

### 3.5. Bàn giao và nghiệm thu

- Dữ liệu đọc được, có manifest và thống kê lớp.
- Không có ảnh trùng chính xác giữa các tập; có báo cáo kiểm tra gần trùng.
- Có nguồn, quy tắc nhãn, quyết định loại mẫu và giới hạn của việc chia tập.
- Người 2 nạp được dữ liệu mà không phải tổ chức lại thủ công.
- Phần báo cáo dữ liệu và hướng dẫn thu ảnh hoàn chỉnh.

**Nội dung phải nắm:** mất cân bằng, stratification, chia theo nhóm, leakage, khác biệt giữa dữ liệu chuẩn và camera, ý nghĩa và giới hạn nhãn `trash`.

## 4. Phần Người 2: Mô hình và huấn luyện

### 4.1. Các thí nghiệm

| Mã | Mô hình | Khởi tạo | Augmentation |
|---|---|---|---|
| E1 | CNN cơ sở | Ngẫu nhiên | Có |
| E2 | ResNet18 | Pretrained ImageNet | Có |
| E3 | MobileNetV3-Small | Pretrained ImageNet | Có |
| E4 | MobileNetV3-Small | Pretrained ImageNet | Không |

E1–E3 so sánh các phương án mô hình. E3–E4 đo tác động của augmentation; mọi thiết lập còn lại phải giống nhau. CNN cơ sở dùng bốn block Conv–BatchNorm–ReLU–MaxPool với số kênh 32/64/128/256, sau đó adaptive average pooling, dropout 0,3 và classifier sáu lớp.

### 4.2. Pipeline chung

1. Nạp manifest đã khóa và kiểm tra sáu nhãn.
2. Thiết lập seed và ghi môi trường chạy.
3. Tạo data loader, kiến trúc và các phép biến đổi ảnh.
4. Huấn luyện, đánh giá validation mỗi epoch, lưu log và checkpoint.
5. Early stopping, lưu checkpoint tốt nhất và checkpoint mới nhất để resume.
6. Xuất đồ thị và thông tin thí nghiệm cho Người 4.

Không viết ba pipeline tách biệt dẫn đến cách tiền xử lý hoặc tính chỉ số khác nhau.

### 4.3. Cấu hình khởi đầu

| Thành phần | Thiết lập |
|---|---|
| Ảnh đầu vào | RGB, 224 × 224 |
| Tiền xử lý đánh giá | Resize cạnh ngắn 256, center crop 224; kiểm tra không cắt mất vật thể |
| Chuẩn hóa | Theo pretrained weights được chọn; ghi cụ thể trong cấu hình/checkpoint |
| Batch size | 32 |
| Optimizer | AdamW, weight decay `1e-4` |
| Loss | Cross-entropy có class weight chỉ tính từ train |
| CNN cơ sở | Tối đa 40 epoch, learning rate ban đầu `1e-3` |
| Pretrained — giai đoạn 1 | Đóng băng backbone, huấn luyện classifier tối đa 8 epoch, learning rate `1e-3` |
| Pretrained — giai đoạn 2 | Mở backbone, fine-tune tối đa 25 epoch, learning rate `1e-4` |
| Chọn checkpoint | Validation macro-F1 cao nhất |
| Early stopping | Patience 5; tính lại ở đầu mỗi giai đoạn |
| Giảm learning rate | ReduceLROnPlateau theo validation macro-F1, mode max, factor 0,5, patience 2 |
| Seed | 42, 123, 2026 |

Augmentation khởi đầu: lật ngang, xoay tối đa 15°, crop có kiểm soát và thay đổi sáng/tương phản nhẹ. Kiểm tra trực quan ảnh sau biến đổi để không làm mất vật hoặc sai nhãn. Giữ BatchNorm của backbone ở chế độ đánh giá khi backbone bị đóng băng.

Ghi rõ trọng số pretrained cụ thể và phiên bản thư viện, không chỉ ghi “pretrained mặc định”. Khi cần điều chỉnh cấu hình, dùng validation và ghi lại các lần thử; không tham khảo test.

### 4.4. Thứ tự chạy và tái lập

1. Smoke test một epoch trên tập nhỏ.
2. Chạy thử E1–E4 với seed 42 để hoàn thiện pipeline và lựa chọn thiết lập bằng validation.
3. Khóa cấu hình; chạy mỗi cấu hình với ba seed 42, 123, 2026.
4. Có tổng cộng 12 lần chạy chính; lần seed 42 chỉ được tái sử dụng nếu đúng cấu hình đã khóa.
5. Ghi cả các lần thử trước khi khóa vào lịch sử, không trộn chúng vào bảng 12 lần chạy chính.
6. Lưu checkpoint định kỳ và hỗ trợ resume để xử lý ngắt phiên Colab.

Seed giúp tái lập quy trình nhưng không bảo đảm kết quả giống tuyệt đối giữa các thiết bị hoặc phiên bản thư viện; ghi môi trường và mọi sai khác đáng kể.

### 4.5. Checkpoint bàn giao

- `state_dict`, tên kiến trúc và tham số cần để dựng lại mô hình.
- Thứ tự sáu nhãn.
- RGB, resize/crop, kích thước ảnh và normalization.
- Cấu hình, seed, epoch, validation metrics và checksum manifest.
- Phiên bản thư viện và định danh pretrained weights.
- Checkpoint resume bổ sung trạng thái optimizer, scheduler và các trạng thái ngẫu nhiên cần thiết.

### 4.6. Nghiệm thu

- Ba kiến trúc chạy được và xuất đúng sáu logits.
- Có log loss, accuracy, macro-F1 theo epoch và cấu hình từng lần chạy.
- Checkpoint tải được trên CPU, không cần tải lại pretrained weights khi demo.
- Kết quả trước và sau lưu/tải checkpoint khớp trong sai số số học cho phép.
- Người 3 chạy được suy luận một ảnh bằng checkpoint đầu tiên từ tuần 2.

**Nội dung phải nắm:** convolution, backbone/classifier, logits/softmax, cross-entropy, gradient, optimizer, learning rate, transfer learning, BatchNorm, overfitting và lựa chọn checkpoint.

## 5. Phần Người 3: Ứng dụng ảnh và camera

### 5.1. Hai chế độ bắt buộc

- Suy luận một ảnh từ đường dẫn.
- Webcam với vùng ROI cố định ở giữa để người dùng đặt một vật thể.

```text
Đọc frame → Cắt ROI → BGR sang RGB → Tiền xử lý
→ Suy luận → Làm mượt → Hiển thị kết quả
```

### 5.2. Chức năng

- Khung hướng dẫn đặt vật; tên lớp, điểm softmax, trạng thái chưa chắc chắn và FPS.
- Nạp nhãn và tiền xử lý từ checkpoint.
- Phím `q` thoát; `s` lưu frame, ROI và kết quả tương ứng.
- Báo rõ khi thiếu checkpoint, sai định dạng, không đọc được ảnh hoặc không mở được camera.
- Chạy offline sau khi cài môi trường và tải đủ tài nguyên.
- Dùng chung hàm tiền xử lý và suy luận với mã đánh giá.

### 5.3. Làm mượt và từ chối

- Lấy trung bình xác suất trong 10 frame suy luận gần nhất.
- Chỉ trả nhãn ổn định khi lớp đó thắng ít nhất 7/10 frame và xác suất trung bình vượt ngưỡng.
- Khi chưa đủ frame hoặc điều kiện không đạt, hiển thị “Chưa chắc chắn”; không giữ nhãn cũ vô thời hạn.
- Ngưỡng 0,65 là giá trị khởi đầu; ngưỡng cuối do Người 4 lựa chọn trên validation và khóa trước đánh giá cuối.
- Lưu ngưỡng và cấu hình smoothing cùng phiên bản ứng dụng.

Ngưỡng confidence không bảo đảm phát hiện nền trống, điện thoại hoặc mọi vật ngoài sáu lớp. Bộ ảnh đánh giá tĩnh không chứng minh hiệu quả smoothing theo thời gian; kiểm tra smoothing riêng bằng chuỗi đổi vật trước camera.

### 5.4. Đo tốc độ

- Báo cáo latency mô hình, thời gian toàn pipeline và FPS thực tế riêng biệt.
- Batch size 1, có warm-up, cùng thiết bị và cùng chế độ suy luận.
- Nếu dùng GPU/MPS bất đồng bộ, đồng bộ thiết bị khi đo thời gian.
- Ghi máy, thiết bị tính toán, kích thước frame, số lượt đo và trung vị/p95 latency.
- Đo thêm độ trễ đổi nhãn sau khi thay vật vì smoothing có thể làm chậm phản hồi.

### 5.5. Nghiệm thu

- Suy luận cùng một ảnh khớp chương trình đánh giá.
- Chạy camera liên tục 10 phút không lỗi.
- Đổi vật thể và bỏ vật ra khỏi khung không gây giữ nhãn cũ vô hạn; ghi nhận trường hợp mô hình vẫn tự tin sai.
- Chạy offline trên máy trình diễn.
- Có hướng dẫn cài đặt, chạy, xử lý lỗi và video dự phòng.

**Nội dung phải nắm:** BGR/RGB, tensor và normalization, `eval`, tắt gradient, ROI, smoothing, latency so với FPS và lỗi chênh tiền xử lý.

## 6. Phần Người 4: Đánh giá độc lập và tổng hợp

### 6.1. Tham gia ngay từ đầu

Người 4 thống nhất tiêu chí, kiểm tra manifest và chuẩn bị mã đánh giá trước khi có mô hình cuối. Test cuối chỉ chạy khi cấu hình, checkpoint lựa chọn và quy tắc ứng dụng đã được khóa bằng validation.

### 6.2. Kết quả bắt buộc

- Accuracy, macro-F1, weighted-F1.
- Precision, recall, F1 và support từng lớp; chỉ số không xác định do không có dự đoán cho lớp được đặt 0 và ghi rõ quy ước.
- Confusion matrix số lượng và chuẩn hóa theo nhãn thật.
- Kết quả từng seed, trung bình và độ lệch chuẩn; độ lệch giữa seed không thay thế khoảng tin cậy lấy mẫu dữ liệu.
- Tham số, kích thước checkpoint, thời gian huấn luyện và tốc độ trên cùng máy demo.
- Bảng riêng cho TrashNet và bộ 300 ảnh thực tế.
- Tối thiểu 12 ảnh sai với nhãn thật, dự đoán, điểm softmax và phân tích nguyên nhân có thể.

### 6.3. Lựa chọn mô hình demo

1. Tính validation macro-F1 trung bình ba seed cho E1–E3.
2. Giữ các phương án cách phương án tốt nhất không quá 0,02 macro-F1, tương đương 2 điểm phần trăm.
3. Trong nhóm này chọn phương án có median latency thấp nhất trên máy demo; nếu hòa, chọn phương án nhỏ hơn về số tham số.
4. Dùng checkpoint seed 42 của phương án được chọn để demo; không chọn seed bằng test.
5. E4 phục vụ ablation, không tự động thay đổi tập ứng viên sau khi đã xem test.

Ghi rõ CNN học từ đầu còn hai mô hình kia được pretrained, nên đây là so sánh phương án triển khai, không phải thí nghiệm chỉ cô lập ảnh hưởng kiến trúc. Ngân sách huấn luyện cũng phải được công bố.

### 6.4. Chọn ngưỡng từ chối

Trên validation của checkpoint demo, quét ngưỡng từ 0,00 đến 0,95 với bước 0,05. Chọn ngưỡng có tỷ lệ chấp nhận cao nhất trong các ngưỡng đạt accuracy trên mẫu chấp nhận từ 90% và coverage từ 70%. Nếu không có ngưỡng đạt cả hai, giữ 0,65 và công bố chưa đạt tiêu chí. Đây là quy tắc vận hành đề xuất, không bảo đảm nhận ra dữ liệu ngoài phân phối.

Báo cáo riêng:

- **Coverage:** số mẫu được chấp nhận / tổng số mẫu.
- **Accuracy trên mẫu chấp nhận:** số mẫu trả nhãn đúng / số mẫu được chấp nhận.
- **Tỷ lệ trả nhãn đúng trên toàn bộ mẫu:** số mẫu được chấp nhận và đúng / tổng số mẫu.

Đồng thời giữ nguyên bảng phân loại sáu lớp không từ chối cho tất cả mô hình. Khi không có mẫu nào được chấp nhận, accuracy trên mẫu chấp nhận ghi N/A. Không bỏ ảnh khó rồi chỉ công bố kết quả trên phần còn lại.

### 6.5. Kiểm thử ngoài phạm vi

Chuẩn bị bộ thử định tính gồm nền trống, tay, điện thoại và nhiều vật thể. Ghi hành vi và lỗi, không gộp vào accuracy sáu lớp hoặc tuyên bố đã giải quyết nhận dạng vật lạ.

### 6.6. Nghiệm thu

- Bảng số liệu sinh từ file dự đoán, không nhập tay.
- Mỗi kết quả truy được về checkpoint, cấu hình, seed và phiên bản dữ liệu.
- Có phân tích domain shift và các nhóm dễ nhầm như paper/cardboard, glass/plastic, trash.
- Kết luận bám sát bằng chứng, ghi rõ các giới hạn và thí nghiệm chưa thực hiện.

**Nội dung phải nắm:** macro/weighted average, confusion matrix, mất cân bằng, ablation, biến thiên seed, công bằng so sánh, đánh giá từ chối và ý nghĩa giới hạn của kết quả.

## 7. Giao diện bàn giao và môi trường

### 7.1. Công cụ và môi trường

- Dùng Git cho mã và cấu hình; dữ liệu/checkpoint lưu riêng, ghi đường dẫn và checksum, không đưa tập dữ liệu lớn vào Git.
- Mac demo dùng môi trường Python 3.12 qua `uv`; kiểm tra cài đặt thực tế trước khi khóa phiên bản thư viện.
- Colab huấn luyện dùng cùng phiên bản PyTorch/torchvision tương thích đã được kiểm chứng; lưu thông tin môi trường trong mỗi lần chạy.
- Đồng bộ checkpoint từ Colab ra nơi lưu bền vững; kiểm tra tải lại sau đồng bộ.
- Người 4 quản lý danh sách thí nghiệm và phiên bản bàn giao, không tự sửa kết quả gốc.

### 7.2. Giao diện dòng lệnh dự kiến

Các lệnh dưới đây là giao diện cần xây dựng, chưa phải chương trình có sẵn:

```bash
python -m src.prepare_data --data-dir data/raw --output data/manifest.csv
python -m src.train --model mobilenet_v3_small --config configs/train.yaml
python -m src.evaluate --checkpoint artifacts/best_model.pt --manifest data/manifest.csv --split test
python -m src.evaluate --checkpoint artifacts/best_model.pt --manifest data/external_test.csv --split test
python -m src.predict --checkpoint artifacts/best_model.pt --image sample.jpg
python -m src.camera --checkpoint artifacts/best_model.pt --camera 0
```

Mỗi lần train/evaluate tạo thư mục kết quả riêng theo mã thí nghiệm và seed. Không ghi đè kết quả cũ mặc định. Evaluation xuất ít nhất `predictions.csv`, `metrics.json`, confusion matrix và cấu hình đã sử dụng.

### 7.3. Các mốc bàn giao

| Mốc | Bàn giao | Điều kiện chuyển bước |
|---|---|---|
| A — Khóa dữ liệu | Người 1 → Người 2, 4 | Manifest, nhãn, thống kê và kiểm tra trùng hoàn tất |
| B — Mô hình đầu tiên | Người 2 → Người 3 | Checkpoint tải được, có nhãn và tiền xử lý |
| C — Khóa thí nghiệm | Người 2 → Người 4 | Cấu hình chọn bằng validation, đủ log và checkpoint |
| D — Tích hợp | Người 3 → Người 4 | Suy luận ảnh khớp đánh giá, camera ổn định |
| E — Đánh giá cuối | Người 4 → Cả nhóm | Đủ bảng kết quả, phân tích lỗi, tốc độ và giới hạn |
| F — Sẵn sàng bảo vệ | Cả nhóm | Báo cáo, slide, demo, dự phòng và hỏi đáp thống nhất |

Nếu làm nối tiếp, ưu tiên A → B → camera cơ bản → C → D → E → F. Người tiếp nhận đọc trước tài liệu đầu vào để phát hiện thiếu sót mà không cần phát triển đồng thời.

## 8. Tiến độ 8 tuần

| Tuần | Người chính | Công việc | Đầu ra bắt buộc |
|---|---|---|---|
| 1 | Người 1, Người 4 | Chốt phạm vi, môi trường, nguồn dữ liệu, EDA, quy tắc nhãn và chia tập | Mốc A; đặc tả bàn giao và kế hoạch thí nghiệm |
| 2 | Người 2 | Data loader, CNN, training loop, log và checkpoint | Baseline end-to-end; mốc B |
| 3 | Người 2, Người 3 | Hai mô hình pretrained, E1–E4 thử nghiệm; suy luận ảnh | Validation sơ bộ; ứng dụng đọc được checkpoint |
| 4 | Người 2 | Khóa cấu hình và chạy ba seed; hoàn thiện log | Mốc C; 12 lần chạy chính hoặc trạng thái và nguyên nhân chạy thiếu |
| 5 | Người 1, Người 3 | Hoàn tất ảnh thực tế, camera, đo tốc độ và sửa lỗi tích hợp | Bộ ảnh được khóa; mốc D |
| 6 | Người 4 | Chọn mô hình bằng validation/tốc độ, khóa ngưỡng rồi chạy test cuối | Mốc E; bảng kết quả và phân tích lỗi |
| 7 | Cả nhóm | Viết báo cáo từng phần, biên tập, slide, README và video | Bản hoàn chỉnh; đóng băng tính năng |
| 8 | Cả nhóm | Cài lại môi trường, demo offline, diễn tập, sửa lỗi còn lại | Mốc F; gói nộp cuối cùng |

Mỗi tuần họp ngắn một lần, ghi ba nội dung: sản phẩm đã có, cách đã kiểm tra, đầu vào còn thiếu. Tuần 8 không thêm tính năng mới. Nếu chậm, ưu tiên pipeline đúng và demo ổn định; mọi lần chạy bị cắt phải được công bố, không điền số liệu thay thế.

## 9. Kiểm thử và điều kiện hoàn thành

### 9.1. Kiểm thử kỹ thuật

- [ ] Manifest có đường dẫn hợp lệ và đủ sáu nhãn; ảnh lỗi được xử lý có nhật ký.
- [ ] Không có ảnh trùng chính xác hoặc nhóm đã biết bị tách giữa train/validation/test.
- [ ] Class weight chỉ tính từ train; validation/test không augmentation ngẫu nhiên.
- [ ] Thứ tự nhãn nhất quán giữa dataset, checkpoint, evaluator và camera.
- [ ] Tiền xử lý đánh giá và ứng dụng cho cùng tensor với cùng đầu vào.
- [ ] Huấn luyện smoke test một epoch; loss và logits hữu hạn.
- [ ] Lưu/tải checkpoint khớp kết quả; tải được trên CPU.
- [ ] Resume hoạt động và không ghi đè log của thí nghiệm khác.
- [ ] Metrics được kiểm tra bằng ví dụ dự đoán nhỏ biết trước kết quả.
- [ ] Thiếu camera/checkpoint/ảnh được báo lỗi rõ ràng.
- [ ] Camera chạy offline 10 phút; đo được tốc độ và độ trễ đổi nhãn.

### 9.2. Kịch bản thực tế

| Kịch bản | Nội dung ghi nhận |
|---|---|
| Một vật rõ, nền sạch | Khả năng nhận đúng và nhãn ổn định |
| Nền phức tạp | Mức suy giảm hoặc trường hợp mô hình dựa vào nền |
| Thiếu sáng, phản sáng | Lỗi phân loại và mức confidence |
| Vật méo, góc lạ | Khả năng tổng quát |
| Khoảng cách thay đổi | Vật còn đủ lớn trong ROI hay không |
| Đổi vật liên tục | Độ trễ smoothing, nhãn cũ và trạng thái chuyển tiếp |
| Không có vật | Hành vi ngoài phạm vi, kể cả dự đoán tự tin sai |
| Điện thoại, tay, nhiều vật | Giới hạn hệ thống sáu lớp một vật thể |

Không dùng các lần demo lặp lại để điều chỉnh mô hình sau khi đã công bố test cuối. Nếu phải sửa lỗi phần mềm ảnh hưởng kết quả, ghi phiên bản sửa và chạy lại toàn bộ đánh giá bị ảnh hưởng.

### 9.3. Checklist gói nộp

- [ ] Mã nguồn và hướng dẫn cài đặt/chạy.
- [ ] Manifest, nguồn dữ liệu, quy tắc nhãn và báo cáo làm sạch.
- [ ] Cấu hình, log, checkpoint và định danh môi trường.
- [ ] Bảng E1–E4, kết quả từng seed, trung bình và độ lệch chuẩn.
- [ ] Đánh giá test TrashNet và 300 ảnh thực tế tách riêng.
- [ ] Confusion matrix, learning curves và ít nhất 12 ảnh sai có phân tích.
- [ ] Kết quả từ chối kèm coverage và tốc độ trên máy demo.
- [ ] Báo cáo, slide, bảng đóng góp, trích dẫn và video dự phòng.

## 10. Báo cáo và bảo vệ

### 10.1. Phân công viết

| Nội dung | Người viết chính |
|---|---|
| Dữ liệu, nhãn, chia tập, ảnh tự chụp | Người 1 |
| CNN, transfer learning, mô hình và huấn luyện | Người 2 |
| Thiết kế ứng dụng, tích hợp, camera và hiệu năng | Người 3 |
| Thực nghiệm, kết quả, phân tích và kết luận | Người 4 |
| Mục tiêu, nghiên cứu liên quan, giới hạn và slide | Cả nhóm; Người 4 biên tập |

### 10.2. Cấu trúc báo cáo

1. Đặt vấn đề, mục tiêu, phạm vi và giả định.
2. Cơ sở lý thuyết và nghiên cứu liên quan.
3. Dữ liệu, quy tắc nhãn, chia tập và kiểm tra chất lượng.
4. Phương pháp, kiến trúc và quy trình huấn luyện.
5. Thiết kế hệ thống suy luận ảnh/camera.
6. Cấu hình và kết quả thực nghiệm.
7. Phân tích lỗi, domain shift và hạn chế.
8. Kết luận, đóng góp thực tế và hướng phát triển.
9. Tài liệu tham khảo, hướng dẫn tái lập và bảng đóng góp.

### 10.3. Câu hỏi mọi thành viên phải trả lời được

1. Đây là classification hay detection? Giả định một vật thể ảnh hưởng thế nào?
2. Tại sao chọn TrashNet và sáu nhãn này? `trash` khác “không xác định” thế nào?
3. Train, validation, test khác nhau ở đâu? Nhóm ngăn leakage như thế nào?
4. Nhóm tái sử dụng gì và tự triển khai gì?
5. Vì sao dùng pretrained? Đóng băng khác fine-tune thế nào?
6. Vì sao macro-F1 cần thiết bên cạnh accuracy?
7. Mô hình nào được chọn cho demo và bằng bằng chứng nào?
8. Tại sao test TrashNet tốt nhưng camera có thể kém?
9. Softmax 90% có bảo đảm 90% khả năng đúng không?
10. Ngưỡng từ chối có hạn chế gì? Có nhận được vật lạ không?
11. FPS và thời gian suy luận được đo như thế nào?
12. Nhóm xử lý dự đoán sai, nhãn nhấp nháy và mất camera ra sao?
13. Vì sao nhiều ảnh của một vật không tương đương nhiều vật độc lập?
14. Kết luận nào được thí nghiệm hỗ trợ, kết luận nào chưa thể đưa ra?

### 10.4. Ba lượt diễn tập

1. Mỗi người trình bày phần mình; thành viên khác hỏi sâu về mã, số liệu và quyết định.
2. Đổi vai: mỗi người giải thích toàn bộ pipeline trong 3 phút.
3. Chạy như buổi báo cáo thật: dùng vật thể mới, tắt Internet và thử phương án dự phòng khi camera lỗi.

Chuẩn bị checkpoint cục bộ, video demo, ảnh mẫu, môi trường chạy đã kiểm tra và một số vật thể an toàn thuộc sáu lớp. Trích dẫn tài liệu đúng nguồn; không đưa kết quả của tác giả khác vào bảng như kết quả nhóm tự chạy.

## 11. Rủi ro và phương án xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Colab hết GPU hoặc ngắt phiên | Checkpoint/resume, lưu bền vững, chạy baseline và cấu hình chính sớm |
| Kết quả ngoài thực tế thấp | Báo cáo domain shift, phân tích nền/góc/ánh sáng; không fine-tune trên test cuối |
| Lớp `trash` ít mẫu và khó định nghĩa | Class weight từ train, kiểm tra nhãn, báo cáo per-class F1 |
| Dữ liệu gần trùng | Kiểm tra hash, ảnh gần trùng và nhóm vật thể; công bố giới hạn còn lại |
| Camera khác kết quả offline | Kiểm tra RGB, crop, normalization, nhãn và chế độ `eval` |
| Smoothing chậm hoặc giữ nhãn cũ | Kiểm tra chuỗi đổi vật, reset/trạng thái chưa chắc chắn và đo độ trễ |
| Môi trường Mac/Colab khác nhau | Kiểm tra checkpoint CPU sớm, ghi phiên bản và kiểm thử máy demo |
| Trễ tiến độ | Đóng băng tính năng, ưu tiên dữ liệu đúng, mô hình chính, đánh giá và demo; ghi rõ phần chưa hoàn thành |

## 12. Nguyên tắc kết luận khoa học

- Mọi con số trong báo cáo phải có đầu ra thực nghiệm truy xuất được.
- Accuracy cao trên TrashNet không đồng nghĩa độ chính xác tương đương ngoài thực tế.
- Confidence softmax chưa hiệu chỉnh không phải bảo đảm xác suất đúng.
- Nhãn vật liệu không đủ để xác định đầy đủ khả năng tái chế hoặc cách xử lý tại địa phương.
- Nhóm cần giải thích hệ thống làm tốt gì, thất bại ở đâu, có bằng chứng nào và cần thí nghiệm gì tiếp theo.

Hoàn thành đề tài được xác nhận bằng sản phẩm hoạt động, phương pháp đánh giá hợp lệ, kết quả minh bạch và khả năng giải thích của cả bốn thành viên.
