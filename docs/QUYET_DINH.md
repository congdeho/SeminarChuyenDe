# Nhật ký quyết định

## 06/10/2026 — Khởi tạo Phase 1

- Một người thực hiện, chia công việc thành sáu phase theo `KE_HOACH_SOLO_6_PHASE.md`.
- Python cục bộ dùng 3.12 vì Python hệ thống là 3.14; mã dự án cho phép 3.12–3.13 để phù hợp runtime Colab đã kiểm tra.
- Dùng `uv` để quản lý môi trường và `uv.lock` để khóa phiên bản sau khi giải được dependencies.
- Thứ tự sáu nhãn chỉ được định nghĩa trong `src/labels.py`.
- Dữ liệu, checkpoint và kết quả chạy để ngoài Git; cấu hình, mã và `uv.lock` được theo dõi bằng Git.
- Chưa tải dữ liệu hoặc chạy thí nghiệm trong Phase 1.
- Môi trường Mac đã khóa bằng `uv.lock`; Colab được kiểm tra riêng với Python 3.13.15, GPU T4 và PyTorch 2.11.0+cu130. Chỉ dùng chung mã nguồn sau khi chạy kiểm thử ở cả hai runtime.
- Notebook Colab cục bộ được lưu trong dự án; do tiện ích Chrome không có quyền đọc tệp cục bộ, bản kiểm tra tương đương được tạo và thực thi trực tiếp trong Colab. Liên kết và kết quả ghi ở `docs/MOI_TRUONG.md`.

## 06/10/2026 — Khóa dữ liệu Phase 2

- Tải ZIP resize trực tiếp từ TrashNet tại commit `6fa2b878c6c1b4304b91109070ce0edf9279bb31`; checksum cố định trong mã để phát hiện nguồn thay đổi.
- Giữ dữ liệu gốc nguyên trạng. Ba cặp SHA giống hệt nhưng khác nhãn được loại cả hai phía khỏi manifest, ghi lại sáu đường dẫn trong báo cáo thay vì đoán nhãn đúng.
- Hai mươi hai cặp gần trùng cùng nhãn theo pHash/dHash được xem trên contact sheet và gộp trước chia tập. Không gán `object_id` giả cho TrashNet; `group_id` là nhóm ứng viên chống rò rỉ, không phải định danh vật thể chắc chắn.
- Chia theo lớp và nhóm xấp xỉ 70/15/15, seed 42. Manifest SHA-256 `eea2338ba07941cd61abff4703734126b0275f4a51bad4d664d09ab4f63dd1de`; train/val/test lần lượt 1.765/378/378.
- Train và validation được nạp riêng; test khóa cho Phase 5. Ảnh camera tự chụp chưa có; quy tắc thu thập đã được chốt trước tại `docs/QUY_TRINH_CHUP_ANH_CAMERA.md`.

## 06/10/2026 — Hoàn thành Phase 3 trên Mac MPS

- Kế hoạch ban đầu dùng Colab T4. Kiểm tra ngoài sandbox cho thấy Mac MPS khả dụng và huấn luyện một epoch đầy đủ ~40 giây, nên hoàn thành run trực tiếp trên Mac; không cần tải dữ liệu lên tài khoản Google để đạt checkpoint. Notebook Colab được chuẩn bị nhưng chưa chạy, và không được gán các chỉ số MPS cho Colab.
- `IMAGENET1K_V1` cho MobileNetV3-Small, thay lớp cuối sáu nhãn. Cấu hình cố định ở `configs/train.yaml`; chọn checkpoint theo validation macro-F1, không mở test.
- Run `local_pretrained_probe` hoàn tất 8 classifier + 10 fine-tune epoch; early stopping fine-tune. Best epoch 13, validation macro-F1 0,8637/accuracy 0,8730; SHA-256 `best.pt` là `083945c828c8dbc7910413f8a15320a00e1425b8e7b8f39156becb0f71942d32`.
- Giữ `runs/` ngoài Git vì chứa checkpoint và metrics sinh ra; cần sao lưu riêng trước khi xóa hoặc chuyển máy. Cũng giữ 6 ảnh mâu thuẫn nhãn ngoài manifest như Phase 2.

## 06/10/2026 — Pipeline camera Phase 4, chưa nghiệm thu webcam

- Dùng đúng `best.pt` Phase 3, chung hàm `predict_pil` cho ảnh và camera; kiểm thử ROI PNG cho sáu điểm khớp dự đoán ảnh lưu.
- Ngưỡng 0,65 và smoothing 7/10 chỉ là quy tắc hiển thị phát triển. Không coi softmax là OOD detector. Thấy ảnh validation `paper/paper10.jpg` bị đoán `plastic` với điểm cao, ghi vào lỗi thay vì chỉnh ngưỡng bằng test.
- Video fixture 120 frame từ validation cho phép kiểm tra pipeline, overlay và benchmark CPU/MPS; ghi rõ không phải webcam hay bài test ngoài miền.
- macOS từ chối Camera với tiến trình Python qua công cụ dù chạy ngoài sandbox. Không reset/quá quyền hệ thống. Chờ quyền Camera cho Terminal trước khi chụp 60 ảnh dev, đo 10 phút và quay demo thật. Không đánh dấu Phase 4 hoàn thành trước các bằng chứng này.

## 18/11/2026 — Thu hẹp phạm vi và bắt đầu Phase 5

- Người thực hiện yêu cầu bỏ chụp camera trực tiếp để đi tiếp. Giữ ứng dụng camera và video validation giả lập như bằng chứng phần mềm, nhưng không tuyên bố đã nghiệm thu webcam, 15 FPS thật hay độ chính xác trên ảnh tự chụp.
- Không thu 60 ảnh dev/300 ảnh final; so sánh E1–E4 và đánh giá TrashNet vẫn giữ quy tắc train/validation/test. Test TrashNet chỉ mở sau khi chọn mô hình, checkpoint và ngưỡng bằng validation.
- Thêm CNN cơ sở và ResNet18, evaluator có hàng rào tệp khóa test, benchmark batch 1 và quét ngưỡng trên validation. Chỉ số E3 seed 42 đã được tái kiểm bằng evaluator; các run khác đang triển khai. Đánh giá domain shift/ảnh thực tế được ghi là không có dữ liệu, không thay bằng video validation lặp frame.

## Mẫu ghi quyết định mới

```text
Ngày:
Quyết định:
Bằng chứng/lý do:
Các phương án đã cân nhắc:
Ảnh hưởng đến dữ liệu, mô hình hoặc báo cáo:
```
