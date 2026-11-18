# Phase 4 — Demo camera: phần đã kiểm tra và việc còn cần làm

Trạng thái 06/10/2026: **pipeline camera đã viết và kiểm thử bằng video validation**, nhưng **chưa nghiệm thu webcam thật**. macOS từ chối quyền Camera đối với tiến trình Python chạy qua công cụ; mình không đặt lại quyền riêng tư. Chưa có 60 ảnh phát triển, thử 10 phút bằng webcam hoặc video demo quay từ webcam.

**Cập nhật 18/11/2026:** theo yêu cầu, tạm bỏ phần chụp webcam trực tiếp và chuyển sang Phase 5 trên TrashNet. Các mục camera thật ở dưới là phần **không thực hiện**, không được mô tả là đã nghiệm thu. Có thể quay lại sau nếu cần demo trực tiếp khi bảo vệ.

## Luồng xử lý

`Webcam BGR → ROI vuông giữa khung (70%) → RGB/PIL → đúng transform trong checkpoint → MobileNetV3-Small eval/inference_mode → softmax sáu lớp → cửa sổ 10 lần dự đoán → nhãn ổn định/không chắc chắn → overlay và log`.

- Ngưỡng phát triển **0,65**, có thể đổi bằng `--threshold`; 7/10 nhãn thắng và **điểm trung bình của nhãn đó trên cả 10 frame** phải đạt ngưỡng. Dưới 10 frame: `warming_up`. Không đạt: `uncertain`, **không giữ nhãn ổn định cũ**. Khi camera đọc lỗi, bộ nhớ làm mượt được xóa và vòng lặp dừng sau ba lần lỗi liên tiếp.
- Giao diện hiển thị nhãn ổn định/điểm, dự đoán thô, số phiếu, pipeline FPS, model latency, ROI và cảnh báo rằng softmax **không phải bộ phát hiện vật lạ**. Đặt trọn một vật vào ROI; vật bị cắt có thể làm sai nhãn.
- `q`/Esc thoát; `s` lưu **ROI PNG chưa gán nhãn** cùng `captures.csv` (tọa độ ROI, sáu điểm, checksum checkpoint); `m` đánh dấu lúc đổi vật để đo độ trễ tới nhãn ổn định khác; `r` xóa cửa sổ làm mượt. Ảnh nhấn `s` là ảnh demo có dự đoán hiển thị, **không nhập tự động** vào bộ 60 ảnh phát triển đã gán nhãn trước khi xem kết quả.
- `predictions.csv` và `summary.json` được lưu theo run trong `runs/` (Git bỏ qua). Video overlay chỉ lưu khi thêm `--record`, không ghi webcam mặc định.

## Cách chạy trên Mac sau khi Camera được cấp quyền

Từ Terminal trong thư mục dự án (không cần Internet):

```bash
uv run --locked python -m src.camera --device cpu
uv run --locked python -m src.camera --device mps --record
uv run --locked python -m src.camera --device cpu --headless --max-seconds 600
```

Lệnh cuối là thử 10 phút; mở `summary.json` mới sinh, xác nhận `stop_reason=duration_limit`, số frame >0 và không có lỗi. Nếu dùng Terminal mà macOS hỏi quyền Camera, chỉ cấp khi bạn muốn ứng dụng Terminal dùng webcam. Trong System Settings → Privacy & Security → Camera, hiện có ChatGPT được bật nhưng **Terminal không nằm trong danh sách**; tiến trình Python qua công cụ vẫn bị từ chối. Không chạy `tccutil reset Camera` để vượt quyền.

Nếu camera khác index 0, thêm `--camera-index 1`. Nếu MPS không khả dụng trong phiên chạy, dùng `--device cpu`; checkpoint CPU đã kiểm thử. Mọi lệnh đều dùng `runs/local_pretrained_probe/best.pt` mặc định. Ngưỡng cuối và checkpoint demo sau so sánh sẽ khóa ở Phase 5.

## Chụp 60 ảnh phát triển đúng quy trình

`src.capture_dev` mở preview **không hiển thị dự đoán**, yêu cầu nhãn và `object_id` trước khi chụp. Mỗi vật tối đa năm ảnh; metadata ở `data/camera_dev/metadata.csv`. Ví dụ cho vật thủy tinh đầu tiên:

```bash
uv run --locked python -m src.capture_dev --label glass --object-id glass_dev_01 --session-id S1 --background ban_go --lighting cua_so
```

Nhấn `s` ở năm góc/điều kiện khác nhau. Chuẩn bị 2 vật khác nhau/lớp × 5 ảnh = 60 ảnh; dùng `glass_dev_02`, v.v. cho vật thứ hai, tương tự năm lớp còn lại. **Không dùng lại các vật này trong 300 ảnh final**. Sau một lượt nghỉ, kiểm tra lại nhãn trước khi xem mô hình dự đoán, điền `label_reviewed` và `review_note`. Xem quy tắc vật liệu và buổi chụp tại [QUY_TRINH_CHUP_ANH_CAMERA.md](QUY_TRINH_CHUP_ANH_CAMERA.md). Hiện số ảnh phát triển thực tế: **0/60**.

## Bằng chứng video giả lập và tốc độ

`src.make_camera_fixture` tạo một video 120 frame, lặp 20 frame cho mỗi ảnh **validation** của sáu lớp; không dùng ảnh test hay webcam. Video nguồn và danh sách ảnh: `runs/camera_fixture/validation_sequence.mp4/.json`. Chạy cùng `src.camera --input-video ... --headless --record` đã tạo video overlay nội bộ, `predictions.csv` và timing:

| Nguồn giả lập, 120 frame | CPU | MPS |
| --- | ---: | ---: |
| Pipeline throughput hiệu dụng | 30,83 FPS | 74,61 FPS |
| Model latency trung vị | 29,52 ms | 6,61 ms |
| Model latency p95 | 33,85 ms | 9,62 ms |
| Tiền xử lý trung vị | 0,61 ms | 1,25 ms |

Nguồn số: `runs/camera_video_cpu_v2/summary.json` và `runs/camera_video_mps_v2/summary.json`. Đây là video được đọc **nhanh nhất có thể, không phải 20 FPS thời gian thực**; kết quả headless không bao gồm `imshow` hoặc độ trễ cấp frame của webcam thật. Lần MPS đầu có cold-start/kết biên kernel lớn (max 445,66 ms), nên dùng cả p50/p95 thay vì chỉ mean. Mốc ≥15 FPS trên webcam thật **chưa được xác nhận**.

`runs/camera_video_mps_v2/demo.mp4` là video minh họa **giả lập**. Kiểm thử tự động xác nhận ROI PNG lưu ra cho cùng sáu điểm với `src.predict` (sai số số học), video xuất được, smoothing không giữ nhãn cũ vô thời hạn. 16/16 kiểm thử toàn dự án qua.

## Lỗi đã thấy và giới hạn

- `paper/paper10.jpg` trong validation (khay giấy ép màu tím) bị mô hình dự đoán `plastic`: ảnh đầy đủ score plastic 0,9195; ROI trong video khoảng 0,98. Đây là **ảnh validation đã biết nhãn**, không phải webcam thực. Nó cho thấy điểm softmax cao vẫn có thể sai, nhất là vật có bề mặt/chất liệu dễ nhầm. Ngưỡng 0,65 không phát hiện được lỗi này.
- Khi video chuyển từ metal sang ảnh paper nói trên, nhãn metal cũ còn hiện ổn định ở ba frame đầu, sau đó chuyển `uncertain` rồi ổn định nhầm `plastic`; smoothing đánh đổi tốc độ phản hồi để giảm nhấp nháy.
- Chưa đo được độ trễ `m` khi đổi **vật thể thật**, chưa kiểm tra nền trống/điện thoại bằng camera, chưa có thử 10 phút hoặc video webcam. Cần tiếp tục sau khi có quyền Camera và vật thể thật; không trình bày video validation lặp frame như bằng chứng nhận dạng thời gian thực.

## Đoạn nháp báo cáo hệ thống

> Hệ thống nhận từng frame BGR từ webcam, cắt ROI vuông ở giữa khung rồi chuyển RGB và dùng đúng tiền xử lý của checkpoint MobileNetV3-Small. Sáu điểm softmax được tính trong chế độ `eval` không gradient. Để giảm nhấp nháy, nhãn chỉ được công bố khi xuất hiện tối thiểu 7/10 lần suy luận gần nhất và có điểm trung bình ≥0,65; các trạng thái còn lại là chờ hoặc chưa chắc chắn. Ảnh ROI có thể lưu cùng tọa độ, điểm và checksum checkpoint để đối chiếu với dự đoán ảnh tĩnh. Ngưỡng này chỉ là quy tắc hiển thị phát triển, không bảo đảm phát hiện vật ngoài sáu lớp. Tốc độ từ video validation giả lập cần được đo lại trên webcam thật trước khi báo cáo kết quả demo.
