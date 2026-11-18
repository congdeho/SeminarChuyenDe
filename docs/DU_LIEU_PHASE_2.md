# Phase 2 — Dữ liệu, chia tập và giới hạn đánh giá

Ngày kiểm tra: 06/10/2026. Đây là kết quả kiểm tra dữ liệu, **chưa phải kết quả huấn luyện**.

## Nguồn và khả năng tái lập

- Tác giả: Gary Thung và Mindy Yang, [TrashNet](https://github.com/garythung/trashnet). Tác giả yêu cầu trích dẫn repository khi sử dụng dữ liệu.
- Tệp dùng: [`data/dataset-resized.zip` tại commit `6fa2b878...`](https://github.com/garythung/trashnet/blob/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip); không trộn ảnh gốc và ảnh resize.
- SHA-256 ZIP: `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`.
- ZIP giải nén không lỗi; 2.527 ảnh JPG đều đọc được, chế độ RGB, kích thước 512×384. Có một tệp `.DS_Store` trong ZIP, không được coi là ảnh.
- Seed chia tập: `42`. Manifest: `data/processed/trashnet_manifest.csv`; SHA-256 hiện tại: `eea2338ba07941cd61abff4703734126b0275f4a51bad4d664d09ab4f63dd1de`.

## Kiểm tra và quyết định xử lý

Không thấy ảnh hỏng. Phát hiện 3 cặp trùng **chính xác theo SHA-256 nhưng khác nhãn**:

| Nhãn 1 | Nhãn 2 | Quyết định |
| --- | --- | --- |
| `glass/glass115.jpg` | `metal/metal91.jpg` | Loại cả hai khỏi manifest |
| `glass/glass176.jpg` | `plastic/plastic152.jpg` | Loại cả hai khỏi manifest |
| `glass/glass389.jpg` | `plastic/plastic332.jpg` | Loại cả hai khỏi manifest |

Ảnh gốc không bị xóa. Danh sách đầy đủ nằm ở `data/reports/issues.csv`; kiểm tra ảnh và SHA tại `data/reports/image_audit.csv`. Không tự chọn một nhãn cho cặp mâu thuẫn vì không có nhãn độc lập đáng tin. Sau loại 6 ảnh, manifest còn **2.521 ảnh**.

Có 22 cặp ảnh **gần trùng cùng nhãn** theo pHash ≤ 5 bit và dHash ≤ 6 bit. Đã xem contact sheet của các cặp: chúng thể hiện cùng vật thể/cảnh rất gần nhau; mã gộp chúng trong một `group_id` để không đi qua nhiều tập. Danh sách: `data/reports/duplicate_candidates.csv`; hình đối chiếu: `data/reports/duplicate_candidates.jpg`. Không có ứng viên gần trùng khác nhãn theo ngưỡng này. Đây chỉ là kiểm tra ứng viên bằng hash, **không chứng minh đã phát hiện mọi lần chụp cùng vật thể**; TrashNet không có `object_id` hoàn chỉnh. Vì vậy cột `object_id` của TrashNet để trống, không giả vờ là mã vật thể thật.

## Phân bố

| Lớp | Ảnh nguồn | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| cardboard | 403 | 282 | 60 | 61 |
| glass | 501 | 349 | 75 | 74 |
| metal | 410 | 286 | 61 | 62 |
| paper | 594 | 416 | 89 | 89 |
| plastic | 482 | 336 | 72 | 72 |
| trash | 137 | 96 | 21 | 20 |
| **Tổng** | **2.527** | **1.765** | **378** | **378** |

Chia xấp xỉ 70/15/15 theo từng lớp, **sau khi gộp nhóm** và trước augmentation. Tổng ba tập là 2.521 vì loại 6 ảnh mâu thuẫn. Cả sáu lớp có ở mỗi tập. Không có hash trùng chính xác hoặc nhóm ứng viên đã biết đi qua các tập. `trash` chỉ có 137 ảnh nguồn nên đây là lớp ít mẫu rõ rệt; nếu dùng trọng số lớp, chỉ tính từ train.

Biểu đồ phân bố: `data/reports/class_distribution.png`; mẫu ảnh sáu lớp: `data/reports/contact_sheet.jpg`; biến đổi train so với eval: `data/reports/augmented_examples.jpg`; tóm tắt máy đọc: `data/reports/summary.json`.

## Quy trình dùng dữ liệu và điểm cần bảo vệ trước thầy

1. `train` dùng cập nhật trọng số; `val` chọn epoch và cấu hình. `test` khóa lại tới Phase 5, chỉ dùng đánh giá cuối, không chọn mô hình dựa vào nó. API Phase 3 gọi `build_train_val_loaders`; `build_test_loader` tách riêng.
2. Ảnh được chuyển RGB, crop về 224×224 và chuẩn hóa theo ImageNet. Train dùng crop, lật ngang, xoay nhỏ, thay đổi sáng/màu; val/test dùng resize + center crop cố định. Không augmentation trước khi chia tập.
3. Cùng seed và dữ liệu tạo lại cùng manifest. Nếu bổ sung/xóa ảnh, đổi nguồn hoặc quy tắc lọc, phải tạo phiên bản manifest mới và chạy lại đánh giá; không so số liệu như thể cùng test set.
4. TrashNet được chụp chủ yếu trên nền trắng, ánh sáng có kiểm soát theo mô tả tác giả. Kết quả test TrashNet không đại diện cho webcam trong môi trường thực. Bộ ảnh camera tự chụp ở [QUY_TRINH_CHUP_ANH_CAMERA.md](QUY_TRINH_CHUP_ANH_CAMERA.md) là phép kiểm thử domain shift riêng.
5. 22 cặp gần trùng đã gộp không thay thế `object_id` thật. Cùng vật thể ở góc/ánh sáng khác nhiều có thể vẫn lọt sang tập khác. Cần ghi giới hạn này trong báo cáo, không khẳng định “không rò rỉ dữ liệu” tuyệt đối.

## Đoạn nháp báo cáo có thể chỉnh sửa

> Nghiên cứu sử dụng TrashNet của Thung và Yang, gồm 2.527 ảnh ở sáu lớp (cardboard, glass, metal, paper, plastic, trash), kích thước 512×384. Chúng tôi kiểm tra khả năng đọc ảnh, SHA-256 và hash cảm nhận. Ba cặp ảnh trùng chính xác nhưng mang nhãn mâu thuẫn được giữ nguyên trong dữ liệu gốc và loại khỏi manifest, còn 2.521 ảnh. Hai mươi hai cặp gần trùng cùng nhãn được gộp trước khi chia tập. Với seed 42, dữ liệu được chia theo lớp và nhóm thành 1.765 train, 378 validation, 378 test. Vì bộ nguồn không cung cấp định danh vật thể đầy đủ, không thể loại trừ hoàn toàn khả năng cùng vật thể ở nhiều tập. Dữ liệu nền trắng cũng khác điều kiện webcam, nên chúng tôi đánh giá riêng trên ảnh camera tự chụp và báo cáo rõ độ chênh giữa hai miền dữ liệu.

Nguồn trích dẫn trong slide/báo cáo: Gary Thung & Mindy Yang, *TrashNet*, https://github.com/garythung/trashnet (truy cập 06/10/2026). Không dùng số accuracy của tác giả như kết quả của dự án này.
