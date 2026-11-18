# KẾ HOẠCH SOLO — 6 PHASE TRIỂN KHAI SEMINAR

**Đề tài:** Xây dựng và đánh giá hệ thống phân loại chất thải rắn từ camera sử dụng học chuyển giao.  
**Người thực hiện:** Một sinh viên, có trợ lý hỗ trợ mã nguồn, kiểm thử, phân tích và tài liệu.  
**Cập nhật:** 06/10/2026.  
**Thời lượng dự kiến:** 6–8 tuần, tùy thời gian học, GPU và thu ảnh; lịch dưới đây dùng 8 tuần, không phải hạn nộp đã xác nhận.  
**Kế hoạch này thay thế:** Phân công nhóm bốn người trong `KE_HOACH_SEMINAR.md`.

## 1. Mục tiêu và cách kiểm soát

### Phạm vi đã chốt

- Phân loại sáu lớp: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.
- Mỗi ảnh hoặc vùng nhận dạng camera chứa một vật thể rác chính.
- TrashNet là dữ liệu chính; không trộn thêm TACO ở phiên bản này.
- PyTorch để huấn luyện; Google Colab GPU cho thí nghiệm; OpenCV và Mac cho demo.
- Ba mô hình: CNN cơ sở, MobileNetV3-Small và ResNet18.
- Đánh giá tác động augmentation, biến thiên giữa seed và khác biệt giữa ảnh TrashNet với ảnh tự chụp.
- Giữ mục tiêu 300 ảnh kiểm thử thực tế; thêm 60 ảnh phát triển camera dùng vật thể riêng.
- Có báo cáo, slide, hướng dẫn chạy, checkpoint và video dự phòng.
- YOLO, phân đoạn ảnh, nhận nhiều vật thể, web app và phần cứng phân loại tự động nằm ngoài phạm vi bắt buộc.

`trash` là nhãn huấn luyện, còn “Chưa chắc chắn” là trạng thái từ chối trả nhãn. Hệ thống không suy ra chắc chắn khả năng tái chế từ vật liệu.

### Thứ tự thực hiện

| Phase | Mục đích | Đầu ra nhìn thấy được | Lịch dự kiến |
|---|---|---|---|
| 1 | Dựng nền dự án | Môi trường kiểm tra được, cấu trúc mã và quy ước thống nhất | Tuần 1, đầu tuần |
| 2 | Chuẩn bị và khóa dữ liệu | Manifest train/validation/test, thống kê và ảnh minh họa | Tuần 1–2 |
| 3 | Huấn luyện mô hình đầu tiên | MobileNetV3 có checkpoint và nhận dạng được một ảnh | Tuần 3 |
| 4 | Đưa mô hình lên camera | Demo webcam hoạt động offline trên Mac | Tuần 4 |
| 5 | So sánh và đánh giá cuối | Bảng thí nghiệm, confusion matrix, đánh giá 300 ảnh tự chụp | Tuần 5–6 |
| 6 | Hoàn thiện và bảo vệ | Gói nộp, báo cáo, slide, video và bộ hỏi đáp | Tuần 7–8 |

**Nguyên tắc:** Có sản phẩm tối thiểu sau Phase 3–4, rồi mở rộng bằng cùng pipeline. Việc có demo sớm không đồng nghĩa đã hoàn thành phần thực nghiệm hoặc bảo vệ.

### Mỗi phase có ba loại việc

1. **Triển khai:** Viết mã, cấu hình và tài liệu cần thiết.
2. **Xác minh:** Chạy kiểm thử, lưu bằng chứng; chỉ hoàn thành khi đạt điều kiện nghiệm thu.
3. **Hiểu:** Bạn giải thích được mục đích, quyết định và kết quả của phần vừa làm.

Không chờ đến tuần cuối mới học lý thuyết hoặc viết báo cáo. Sau mỗi phase, bổ sung nhật ký và một bản nháp phần báo cáo tương ứng.

### Phân biệt phần trợ lý và phần bạn thực hiện

| Trợ lý hỗ trợ | Bạn cần trực tiếp thực hiện hoặc xác nhận |
|---|---|
| Viết mã, notebook, kiểm thử, cấu hình, hướng dẫn | Chạy notebook trong tài khoản Colab khi cần đăng nhập/cấp GPU |
| Đọc log, tìm lỗi, phân tích số liệu đã có | Cung cấp đầu ra/log nếu phiên huấn luyện không truy cập được từ môi trường này |
| Tạo hướng dẫn và công cụ thu ảnh | Chuẩn bị vật thể, chụp ảnh và xác nhận nhãn thật |
| Viết ứng dụng và kiểm tra phần mềm | Cho phép camera trên macOS và thao tác vật thể khi demo |
| Soạn nháp báo cáo, câu hỏi và giải thích | Kiểm tra nội dung, luyện trình bày và đáp ứng quy định môn học về công cụ hỗ trợ |

Không đánh dấu các bước Colab, camera hoặc chụp ảnh là hoàn tất chỉ vì đã viết xong mã.

## 2. Phase 1 — Nền dự án và môi trường

**Mục tiêu:** Từ thư mục chứa kế hoạch thành dự án có môi trường và cách chạy nhất quán. Chưa huấn luyện mô hình thật.

**Đầu vào:** Kế hoạch, máy Mac và lựa chọn Colab đã thống nhất.

### Công việc nhỏ

- [x] P1.1 — Kiểm tra lại Python, `uv`, CPU/MPS và các công cụ hiện có; ghi thông tin máy demo.
- [x] P1.2 — Tạo cấu trúc `src`, `configs`, `notebooks`, `tests`, `docs`; quy ước thư mục dữ liệu, checkpoint và kết quả.
- [x] P1.3 — Tạo cấu hình môi trường Python 3.12 cục bộ và cài nhóm thư viện PyTorch, torchvision, OpenCV, xử lý ảnh, metrics và biểu đồ.
- [x] P1.4 — Kiểm tra cài đặt, khóa phiên bản tương thích sau khi chạy thử; ghi rõ môi trường Colab có thể khác Python cục bộ.
- [x] P1.5 — Tạo `.gitignore` và khởi tạo Git cục bộ; không đưa dữ liệu, checkpoint, token và môi trường ảo vào Git. Chưa tạo remote hoặc đẩy mã lên dịch vụ ngoài.
- [x] P1.6 — Cố định nhãn theo thứ tự `cardboard, glass, metal, paper, plastic, trash`; tạo cấu hình chung và đường dẫn tương đối.
- [x] P1.7 — Tạo công cụ kiểm tra môi trường, README khởi động và notebook Colab tối thiểu kiểm tra runtime/GPU.
- [x] P1.8 — Lập nhật ký quyết định và sổ kết quả thí nghiệm chưa có số liệu.

### Đầu ra và nghiệm thu

- Môi trường import được các thư viện cần thiết và kiểm tra được CPU/MPS; MPS không phải điều kiện bắt buộc nếu CPU chạy được demo.
- Công cụ kiểm tra môi trường chạy được bằng lệnh trong README; notebook Colab có đầu ra kiểm tra runtime thực tế trước khi ghi nhận Colab sẵn sàng.
- Thông tin sáu nhãn chỉ có một nguồn dùng chung.
- README phân biệt rõ lệnh đã chạy được và lệnh sẽ triển khai ở phase sau.
- Không có tuyên bố đã huấn luyện hay số liệu accuracy khi chưa chạy.

**Bạn cần hiểu:** Environment là gì; train và inference khác nhau thế nào; checkpoint chứa gì; vai trò Mac và Colab trong dự án.

**Bằng chứng lưu:** Kết quả kiểm tra môi trường, phiên bản thư viện và nhật ký khởi tạo.

## 3. Phase 2 — Dữ liệu sạch và cách đánh giá

**Mục tiêu:** Có bộ dữ liệu và cách chia tập đáng tin trước khi huấn luyện.

**Đầu vào:** Phase 1 đạt nghiệm thu.

### Công việc nhỏ

- [x] P2.1 — Xác minh nguồn tải TrashNet của tác giả, tải dữ liệu, ghi nguồn/phiên bản/checksum; đếm lại ảnh thực tế.
- [x] P2.2 — Kiểm tra ảnh hỏng, ảnh trùng chính xác, ảnh gần trùng và nhãn đáng ngờ; lưu danh sách cần xem thay vì âm thầm xóa dữ liệu gốc.
- [x] P2.3 — Thống kê lớp, kích thước ảnh, tạo contact sheet và biểu đồ EDA.
- [x] P2.4 — Xác định nhóm cùng vật thể/gần trùng khi có căn cứ; chia khoảng 70/15/15, seed 42, giữ nhóm không đi qua các tập.
- [x] P2.5 — Khóa manifest `path,label,split,object_id,source`; lưu checksum và phân bố lớp sau chia.
- [x] P2.6 — Tạo Dataset/DataLoader, xử lý RGB, train transforms và phép biến đổi đánh giá xác định.
- [x] P2.7 — Kiểm tra nhãn, shape tensor, augmentation và việc không đọc test trong quy trình chọn cấu hình.
- [x] P2.8 — Viết quy tắc chụp/gán nhãn, phân biệt 60 ảnh phát triển và 300 ảnh kiểm thử cuối; lập danh sách vật thể cần chuẩn bị. (Ảnh tự chụp sẽ thu dần.)
- [x] P2.9 — Viết nháp phần dữ liệu, nguồn và giới hạn của cách chia tập.

### Quy tắc dữ liệu

- Chia tập trước augmentation. Tính class weight chỉ từ train.
- Giữ nhóm vật thể trước, chấp nhận tỷ lệ chia lệch nhẹ và báo cáo số lượng thật.
- Không khẳng định đã loại hết rò rỉ cùng vật thể nếu nguồn không có định danh đầy đủ.
- Với ảnh tự chụp, `object_id` là bắt buộc. Bộ phát triển camera và bộ test dùng vật thể khác nhau.
- 300 ảnh test gồm 50 ảnh/lớp; mục tiêu 10 vật thể/lớp × 5 góc/điều kiện, qua ít nhất 3 buổi.
- Một người gán nhãn rồi kiểm tra lại ở lượt riêng trước khi xem dự đoán; ghi rõ đây là tự kiểm tra, không phải hai người gán nhãn độc lập.
- Vật nhiều chất liệu hoặc nhãn không chắc được ghi thành trường hợp ngoài phạm vi theo quy tắc đặt trước.
- Có thể thu ảnh dần từ Phase 2 đến đầu Phase 5. Chưa cần chụp đủ 300 ảnh để qua Phase 2.

### Đầu ra và nghiệm thu

- Manifest hợp lệ, ảnh đọc được, sáu lớp có mặt trong mỗi tập và không có ảnh trùng chính xác giữa các tập.
- Các nhóm đã biết không bị tách; có báo cáo giới hạn kiểm tra gần trùng.
- Chạy lại bằng cùng dữ liệu/seed tái tạo cùng manifest; thay đổi dữ liệu phải đổi phiên bản.
- DataLoader trả batch đúng; ảnh sau augmentation vẫn thể hiện vật thể và nhãn hợp lệ.
- Có quy tắc ảnh thực tế đủ để bạn chụp theo mà không cần quyết định lại.

**Bạn cần hiểu:** Train/validation/test; class imbalance; stratification; chia theo nhóm; leakage; lý do 300 ảnh không tương đương 300 vật thể độc lập.

**Bằng chứng lưu:** Manifest, checksum, EDA, danh sách xử lý ảnh, mẫu batch và báo cáo kiểm tra chia tập.

## 4. Phase 3 — Mô hình đầu tiên chạy được

**Mục tiêu:** Huấn luyện MobileNetV3-Small trên TrashNet và dự đoán được một ảnh chưa dùng để cập nhật trọng số. Chỉ kiểm tra bằng validation hoặc bộ phát triển, chưa mở test cuối.

**Đầu vào:** Manifest và DataLoader đã khóa ở Phase 2.

### Công việc nhỏ

- [x] P3.1 — Tạo MobileNetV3-Small pretrained ImageNet, thay classifier sáu lớp; ghi định danh trọng số pretrained cụ thể.
- [x] P3.2 — Tạo training loop dùng chung: train, validation, checkpoint, history và resume.
- [x] P3.3 — Smoke test một epoch trên tập nhỏ để kiểm tra loss, gradient, shape và lưu/tải mô hình.
- [x] P3.4 — Tạo notebook Colab gọi mã nguồn dùng chung, lưu kết quả ra vùng lưu bền vững; không sao chép logic huấn luyện riêng vào notebook. (Notebook đã tạo, chưa thực thi trên Colab.)
- [x] P3.5 — Chạy seed 42, huấn luyện classifier rồi fine-tune; chỉ theo dõi validation để quyết định. (Thực hiện trên **Mac MPS thay cho Colab T4**; xem nhật ký thay đổi bên dưới.)
- [x] P3.6 — Xuất learning curves, validation metrics và checkpoint tốt nhất.
- [x] P3.7 — Tạo lệnh dự đoán một ảnh cục bộ, nạp được checkpoint trên CPU và không cần tải pretrained weights lần nữa.
- [x] P3.8 — Kiểm tra resume và so sánh dự đoán trước/sau lưu mô hình.
- [x] P3.9 — Viết nháp phần MobileNetV3, transfer learning và giải thích các siêu tham số đã dùng.

### Cấu hình ban đầu

- RGB 224 × 224; normalization theo weights đã chọn, ghi số cụ thể vào checkpoint.
- Batch size 32, AdamW, weight decay `1e-4`, cross-entropy có class weight từ train.
- Classifier: tối đa 8 epoch, learning rate `1e-3`, backbone đóng băng và BatchNorm backbone ở chế độ đánh giá.
- Fine-tune: mở backbone, tối đa 25 epoch, learning rate `1e-4`.
- Early stopping patience 5 theo validation macro-F1; reset theo giai đoạn; lưu checkpoint tốt nhất toàn bộ quá trình.
- ReduceLROnPlateau: mode max, factor 0,5, patience 2; ghi cách khởi tạo lại optimizer/scheduler khi chuyển giai đoạn.
- Train augmentation: lật ngang, xoay nhẹ tối đa 15°, crop có kiểm soát, thay đổi sáng/tương phản nhẹ.
- Validation/inference: cùng phép biến đổi xác định, mặc định resize cạnh ngắn 256 rồi center crop 224; kiểm tra ảnh thực tế trước khi khóa phép biến đổi.

  ### Đầu ra và nghiệm thu

- Có checkpoint thực sự huấn luyện, history, cấu hình, seed và checksum manifest; không chỉ có pretrained model chưa fine-tune.
- Dự đoán một ảnh chạy trên Mac; trả đủ sáu điểm, thứ tự nhãn đúng.
- Checkpoint nạp lại cho dự đoán khớp trong sai số số học; resume không làm mất lịch sử.
- Một lần train hoàn chỉnh có đủ log, kể cả khi chỉ số chưa cao; chỉ số thấp cần ghi nguyên nhân và kế hoạch kiểm tra.
- Test TrashNet và 300 ảnh test thực tế chưa tham gia chọn tham số.

**Bạn cần hiểu:** Backbone/classifier, transfer learning, logits và softmax, cross-entropy, gradient, AdamW, learning rate, overfitting, early stopping.

**Bằng chứng lưu:** Notebook/output thực tế, checkpoint, history, đồ thị validation, dự đoán mẫu và kiểm thử lưu/tải.

## 5. Phase 4 — Demo camera tối thiểu ổn định

**Mục tiêu:** Đưa checkpoint Phase 3 vào webcam, biết rõ hệ thống đang làm được gì và còn lỗi gì.

**Đầu vào:** Checkpoint và lệnh suy luận ảnh chạy được.

### Công việc nhỏ

- [x] P4.1 — Viết vòng đọc webcam và ROI trung tâm, xử lý không mở được camera. (Mã xong; webcam thật bị macOS từ chối quyền.)
- [x] P4.2 — Chuyển BGR sang RGB, dùng chung tiền xử lý và hàm suy luận với dự đoán ảnh.
- [x] P4.3 — Hiển thị nhãn, điểm softmax, FPS và trạng thái chưa chắc chắn. (Kiểm tra qua video validation giả lập.)
- [x] P4.4 — Làm mượt 10 lần suy luận gần nhất; yêu cầu lớp thắng ít nhất 7/10 lần và vượt ngưỡng điểm trung bình.
- [x] P4.5 — Dùng ngưỡng phát triển 0,65, giữ khả năng cấu hình; ngưỡng cuối sẽ chọn ở Phase 5.
- [x] P4.6 — Thêm phím thoát/lưu frame và metadata; xử lý chưa đủ frame, mất camera và đổi vật. (Mã/logic kiểm thử; phím trên webcam thật chưa nghiệm thu.)
- [ ] P4.7 — Chụp đủ 60 ảnh phát triển, thử nền, ánh sáng, góc và khoảng cách; ghi lỗi. (0/60; đã có công cụ chụp không hiển thị dự đoán.)
- [ ] P4.8 — Đo latency mô hình, toàn pipeline, FPS và độ trễ đổi nhãn; kiểm tra chạy offline 10 phút. (Đã đo video giả lập CPU/MPS; chưa đo webcam 10 phút/đổi vật thật.)
- [ ] P4.9 — Viết nháp phần hệ thống và lưu video demo nội bộ đầu tiên. (Đã viết nháp và video giả lập; chưa có video webcam thật.)

### Đầu ra và nghiệm thu

- Camera dùng đúng checkpoint/nhãn; cùng ROI xuất ra ảnh cho kết quả khớp chế độ ảnh khi chưa smoothing.
- Không giữ nhãn cũ vô thời hạn khi điều kiện ổn định không đạt; hiển thị trạng thái chuyển tiếp rõ ràng.
- Chạy offline 10 phút không lỗi và có số đo trên máy thật, không dùng số tốc độ của nhà cung cấp mô hình.
- Mục tiêu 15 FPS; nếu chưa đạt, ghi kết quả thật và xác định chi phí đọc frame/tiền xử lý/suy luận/hiển thị.
- Có danh sách lỗi ngoài phạm vi như nền trống hoặc điện thoại bị nhận thành rác. Không coi ngưỡng softmax là bộ phát hiện mọi vật lạ.

**Bạn cần hiểu:** BGR/RGB, normalization, `eval`, tắt gradient, ROI, FPS/latency, lý do smoothing làm chậm phản hồi.

**Bằng chứng lưu:** Video thử, số đo tốc độ, kết quả 10 phút, ảnh phát triển, nhật ký lỗi.

## 6. Phase 5 — So sánh mô hình và đánh giá cuối

**Điều chỉnh phạm vi 18/11/2026 theo yêu cầu:** chuyển sang Phase 5 mà không chụp webcam trực tiếp. P4.7–P4.9 và phần 300 ảnh thực tế ở P5.6/P5.7 không được xem là hoàn thành; báo cáo ghi “không có dữ liệu ảnh tự chụp”, không suy ra domain shift hoặc FPS webcam. Tiếp tục nhánh TrashNet và chỉ mở test TrashNet sau khi khóa lựa chọn bằng validation. Chi tiết tiến độ và lệnh ở [docs/PHASE_5.md](docs/PHASE_5.md).

**Mục tiêu:** Chuyển từ demo chạy được sang đề tài có bằng chứng thực nghiệm.

**Đầu vào:** Pipeline huấn luyện và camera đã hoạt động; có quy tắc thu ảnh kiểm thử.

### Công việc nhỏ

- [x] P5.1 — Bổ sung CNN cơ sở và ResNet18 vào cùng pipeline; kiểm thử đầu ra sáu lớp.
- [x] P5.2 — Hoàn thành bốn cấu hình E1–E4 với seed 42 và đánh giá validation.
- [x] P5.3 — Khóa cấu hình và chạy đủ 12 lượt cho ba seed 42/123/2026. Checkpoint E3 seed 42 khớp cấu hình và manifest.
- [x] P5.4 — Đo tốc độ trên Mac MPS và chọn ResNet18 bằng validation và latency.
- [x] P5.5 — Khóa checkpoint ResNet18 seed 42, preprocessing, ngưỡng 0,60 và smoothing 7/10 trước test.
- [ ] P5.6 — Hoàn tất 300 ảnh test, tự kiểm tra lại nhãn trước khi xem dự đoán và khóa manifest. (**Bỏ nhánh này theo điều chỉnh phạm vi**; không gọi là đã làm.)
- [x] P5.7 — Chạy 378 ảnh test TrashNet cho bốn checkpoint seed 42, lưu dự đoán, metrics và confusion matrix. Test ảnh tự chụp đã bỏ theo điều chỉnh phạm vi.
- [x] P5.8 — Tổng hợp ba seed, ablation, coverage, tốc độ và phân tích 12 ảnh test sai. Domain shift ảnh tự chụp ngoài phạm vi hiện tại.
- [x] P5.9 — Kiểm tra checkpoint cuối trên chuỗi video offline 120 frame. Chất lượng webcam thật nằm ngoài phạm vi hiện tại.
- [x] P5.10 — Hoàn thiện kết quả và giới hạn trong báo cáo Phase 6.

### Ma trận thí nghiệm

| Mã | Phương án | Augmentation | Số lần chạy chính |
|---|---|---|---:|
| E1 | CNN cơ sở, khởi tạo ngẫu nhiên | Có | 3 seed |
| E2 | ResNet18 pretrained | Có | 3 seed |
| E3 | MobileNetV3-Small pretrained | Có | 3 seed |
| E4 | MobileNetV3-Small pretrained | Không | 3 seed |

Tổng cộng 12 lần chạy chính; các lần dò cấu hình trước đó ghi riêng. E3 từ Phase 3 chỉ tính vào bảng chính nếu đúng cấu hình đã khóa. E3/E4 giữ nguyên mọi yếu tố ngoài augmentation. CNN gồm bốn block Conv–BatchNorm–ReLU–MaxPool, số kênh 32/64/128/256, adaptive average pooling, dropout 0,3 và linear sáu lớp; tối đa 40 epoch, learning rate ban đầu `1e-3`.

### Chọn mô hình và ngưỡng

1. So sánh validation macro-F1 trung bình ba seed của E1–E3.
2. Trong các kiến trúc cách tốt nhất không quá 0,02 macro-F1, chọn median latency thấp nhất; nếu hòa chọn ít tham số hơn.
3. Dùng checkpoint seed 42 của kiến trúc được chọn cho demo; không chọn seed đẹp nhất bằng test.
4. Quét ngưỡng validation 0,00–0,95, bước 0,05. Trong các ngưỡng có accepted accuracy ≥ 90% và coverage ≥ 70%, chọn coverage cao nhất; nếu hòa chọn ngưỡng thấp nhất. Nếu không đạt, giữ 0,65 và báo cáo chưa đạt tiêu chí này.
5. Ngưỡng từ ảnh validation là quy tắc cho ảnh tĩnh; hiệu quả smoothing trên video cần báo cáo bằng kiểm thử chuỗi camera riêng, không suy ra từ ảnh tĩnh.

### Kết quả phải báo cáo

- Accuracy, macro-F1, weighted-F1, precision/recall/F1/support từng lớp.
- Confusion matrix số lượng và chuẩn hóa theo nhãn thật.
- Kết quả từng seed, trung bình và độ lệch chuẩn; không coi độ lệch giữa seed là khoảng tin cậy của toàn bộ dữ liệu.
- Bảng TrashNet và ảnh thực tế tách riêng; thống kê số ảnh lẫn số vật thể tự chụp.
- Bảng sáu lớp không từ chối cho mọi mô hình; với checkpoint demo thêm coverage, accuracy trên mẫu chấp nhận và tỷ lệ trả nhãn đúng trên toàn bộ ảnh.
- Accuracy trên mẫu chấp nhận ghi N/A nếu không có mẫu được chấp nhận.
- Batch size 1, warm-up và đồng bộ GPU/MPS khi đo; báo cáo median/p95 latency, FPS, số tham số và kích thước checkpoint.
- So sánh CNN từ đầu với pretrained được diễn giải là so sánh phương án, không chỉ riêng kiến trúc.

### Đầu ra và nghiệm thu

- Mọi hàng trong bảng truy được về cấu hình, seed, checkpoint và manifest; số liệu sinh từ dự đoán, không nhập tay.
- Có kiểm thử evaluator bằng tập dự đoán nhỏ biết trước đáp án.
- Test chỉ chạy sau khi khóa các lựa chọn; lỗi phần mềm làm sai kết quả phải ghi bản sửa và chạy lại các đánh giá bị ảnh hưởng.
- Có đủ ba mô hình, ablation, đánh giá ảnh thật và phân tích lỗi. Thiếu lượt chạy phải nêu rõ, không điền số liệu hoặc gọi một seed là kết quả ba seed.
- Mục tiêu accuracy 85% và macro-F1 0,80 trên TrashNet là mục tiêu tham khảo; không phải điều kiện để sửa hoặc lựa lại test.

**Bạn cần hiểu:** Macro/weighted-F1, confusion matrix, ablation, công bằng thí nghiệm, seed, domain shift, overconfidence và giới hạn của bộ test nhỏ.

**Bằng chứng lưu:** Predictions, metrics, cấu hình đã khóa, bảng số liệu, hình ảnh sai, benchmark và kết quả camera cuối.

## 7. Phase 6 — Đóng gói và luyện bảo vệ

**Mục tiêu:** Một người khác chạy lại được dự án và bạn giải thích được các quyết định quan trọng.

### Công việc nhỏ

- [x] P6.1 — Hoàn thiện README, cài đặt, tải dữ liệu/checkpoint, các lệnh và lỗi thường gặp. (Bản luyện có gói offline và hướng dẫn đã chạy lại.)
- [x] P6.2 — Hoàn thiện báo cáo phương pháp, nguồn tham khảo và kết quả đã khóa.
- [x] P6.3 — Cập nhật tám slide với kết quả validation ba seed và test.
- [x] P6.4 — Xuất video offline 120 frame, kèm ảnh mẫu và checkpoint ResNet18 seed 42.
- [x] P6.5 — Cài vào môi trường mới, kiểm tra lệnh cốt lõi và suy luận offline. (Đã kiểm tra bằng bản giải nén với CPU và video dựng từ ảnh validation. Webcam thật ngoài phạm vi hiện tại.)
- [x] P6.6 — Soạn câu hỏi phản biện gắn với mã/biểu đồ của chính dự án.
- [ ] P6.7 — Tự trình bày pipeline trong 3 phút, báo cáo thử đủ thời lượng môn học và luyện hỏi đáp.
- [x] P6.8 — Lưu bản kết quả ở `outputs/phase6/ban_ket_qua/` cùng evidence và checksum.

### Đầu ra và nghiệm thu

- Báo cáo, slide, README, mã nguồn, cấu hình, checkpoint và kết quả không mâu thuẫn nhau.
- Có hướng dẫn tái lập với điều kiện GPU/thời gian, không hứa kết quả bitwise giống hệt giữa phần cứng khác nhau.
- Demo trên máy báo cáo khi tắt mạng; video dự phòng mở được.
- Bạn chỉ ra được phần tự làm, phần tái sử dụng và cách dùng công cụ hỗ trợ theo quy định của môn.
- Bạn giải thích được ít nhất một trường hợp sai và giới hạn của mọi kết luận chính.

### Câu hỏi bắt buộc luyện

1. Vì sao chọn classification một vật thay vì detection nhiều vật?
2. Sáu lớp có ý nghĩa gì, `trash` và unknown khác nhau thế nào?
3. Cách chia dữ liệu ngăn rò rỉ ra sao, còn giới hạn gì?
4. Vì sao dùng pretrained và fine-tune? Đóng băng backbone ảnh hưởng BatchNorm thế nào?
5. Vì sao chọn checkpoint bằng validation macro-F1?
6. MobileNet/ResNet/CNN khác nhau ở điểm nào liên quan đến kết quả của bạn?
7. Augmentation cải thiện bao nhiêu trong thí nghiệm thực sự?
8. Độ chính xác giảm bao nhiêu trên ảnh thật, có thể do đâu?
9. Softmax cao có bảo đảm đúng không, ngưỡng từ chối có hạn chế gì?
10. FPS và latency được đo thế nào, smoothing thêm độ trễ bao nhiêu?
11. Khi webcam chỉ thấy nền hoặc điện thoại thì hệ thống làm gì?
12. Nếu có thêm thời gian/dữ liệu, bạn sẽ kiểm chứng cải tiến nào tiếp theo?

## 8. Quy ước giao diện và lưu kết quả

Các lệnh là giao diện dự kiến; chỉ đánh dấu khả dụng khi phase tương ứng đã triển khai và kiểm thử:

```text
check_env     → thông tin môi trường và kiểm tra thiết bị
prepare_data  → manifest, checksum và báo cáo dữ liệu
train         → checkpoint, cấu hình và lịch sử huấn luyện
predict       → nhãn và sáu điểm của một ảnh
camera        → ứng dụng webcam
evaluate      → predictions, metrics và confusion matrix
benchmark     → latency/FPS với thông tin thiết bị
```

Checkpoint chứa kiến trúc, trọng số, nhãn, preprocessing, cấu hình, seed, epoch, validation metrics, checksum manifest và phiên bản thư viện. Checkpoint resume thêm optimizer/scheduler và trạng thái ngẫu nhiên cần thiết. Các lần chạy lưu riêng theo mã thí nghiệm/seed, không ghi đè mặc định.

## 9. Bảng theo dõi hiện tại

Kiểm tra ngày 18/11/2026. Phase 1–3 hoàn thành. Phase 4 đã có phần mềm và video dựng từ validation. Phase 5 hoàn thành trên TrashNet với 12 lượt huấn luyện và tệp khóa ResNet18 seed 42. Trên 378 ảnh test, checkpoint đạt accuracy 88,89% và macro-F1 87,02%. Phase 6 có báo cáo, slide và gói demo offline. Webcam thật và ảnh tự chụp nằm ngoài phạm vi đã điều chỉnh.

| Phase | Trạng thái | Bằng chứng hiện có | Bước kế tiếp |
|---|---|---|---|
| 1 | Hoàn thành | Python 3.12, `uv.lock`, mã kiểm tra, Git, Colab T4 đã chạy | Khóa kết quả và dùng làm nền |
| 2 | Hoàn thành | ZIP/checksum, audit, EDA, manifest, DataLoader, quy trình ảnh camera | Thu ảnh camera dần; không mở test để chọn mô hình |
| 3 | Hoàn thành trên Mac MPS | Run 18 epoch, `best.pt`, history/curves/validation report, dự đoán CPU, resume | Giữ test khóa; Colab notebook là phương án tái chạy chưa kiểm chứng |
| 4 | Dừng nhánh webcam thật theo yêu cầu | Mã webcam, ROI/smoothing/save/log, video validation giả lập và benchmark CPU/MPS | Không báo cáo như nghiệm thu camera thật |
| 5 | Hoàn thành trên TrashNet | 12 lượt chạy, tệp khóa, dự đoán và metrics test, 12 ảnh sai, benchmark và kết quả video | Bảo toàn tệp khóa và dữ liệu gốc |
| 6 | Có bản kết quả | Báo cáo, PowerPoint 8 slide, gói demo offline và 12 câu phản biện | Người trình bày tập bảo vệ |

### Mẫu cập nhật cuối mỗi phiên làm việc

```text
Phase và bước đang làm:
Việc đã thực hiện:
Tệp/đầu ra mới:
Kiểm tra đã chạy và kết quả:
Điều bạn cần hiểu:
Việc cần bạn thao tác (nếu có):
Lỗi hoặc hạn chế còn lại:
Bước tiếp theo:
```

Chỉ cập nhật “Hoàn thành” khi có đầu ra và kiểm tra nghiệm thu. Không yêu cầu phê duyệt lại cho từng thao tác kỹ thuật thông thường; khi cần tài khoản, camera, vật thể thật hoặc một quyết định đổi phạm vi, ghi rõ phần phụ thuộc và tiếp tục phần có thể làm độc lập.

## 10. Kiểm soát khối lượng khi làm một mình

- Giữ nguyên sáu phase và bộ sản phẩm chính; không bổ sung giao diện web hoặc YOLO giữa chừng.
- Tập trung MobileNet trước để phát hiện sớm lỗi dữ liệu/huấn luyện/tích hợp, rồi tái sử dụng pipeline cho các mô hình còn lại.
- Thu ảnh theo danh sách vật thể và lưu `object_id` ngay từ đầu; tránh phải chụp lại vì trộn vật thể phát triển/test.
- Lưu log và viết nháp báo cáo ngay khi có kết quả; không tái tạo số liệu từ trí nhớ.
- Nếu GPU/thời gian thiếu, ưu tiên hoàn tất bốn cấu hình seed 42 trước. Việc giảm tổng số seed là thay đổi phạm vi phải ghi rõ trước đánh giá test, báo cáo giới hạn và điều chỉnh cách chọn mô hình; không âm thầm hạ chuẩn.
- Khi đạt nghiệm thu phase, chuyển bước; chỉ quay lại khi có lỗi hoặc bằng chứng mới cần sửa. Không tối ưu vô hạn để đuổi theo một con số accuracy.

**Bước chuẩn bị bảo vệ.** Dùng báo cáo, slide và video trong bản kết quả. Người trình bày điền thông tin cá nhân và luyện theo thời lượng của môn.
