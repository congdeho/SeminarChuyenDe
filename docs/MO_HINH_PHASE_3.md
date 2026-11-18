# Phase 3 — MobileNetV3-Small và nhật ký triển khai

Trạng thái 06/10/2026: **đã huấn luyện đầy đủ trên Mac MPS**, thay cho T4 trong kế hoạch vì máy hiện tại chạy MPS tốt. Notebook Colab đã chuẩn bị nhưng chưa chạy; không được nói rằng kết quả sau đây đến từ Colab. Không dùng số từ smoke test như kết quả seminar.

## Mô hình và quy trình

- `torchvision.models.mobilenet_v3_small` với trọng số `MobileNet_V3_Small_Weights.IMAGENET1K_V1` tải từ nguồn PyTorch chính thức (`mobilenet_v3_small-047dcff4.pth`). Trọng số pretrained 9,83 MiB đã tải vào cache cục bộ. Lớp cuối được thay bằng `Linear(1024, 6)` theo thứ tự nhãn trong `src/labels.py`. Mô hình sau thay có 1.524.006 tham số (đếm trên runtime cục bộ).
- Ảnh RGB, resize cạnh ngắn 256, center crop 224, chuẩn hóa mean `(0.485, 0.456, 0.406)` và std `(0.229, 0.224, 0.225)` ở validation/inference. Train dùng augmentation đã chốt ở Phase 2. Tiền xử lý và thứ tự nhãn được lưu trong checkpoint.
- Giai đoạn `classifier`: đóng băng `features` và giữ BatchNorm backbone ở chế độ eval; tối đa 8 epoch, AdamW lr `1e-3`. Giai đoạn `finetune`: mở backbone, tối đa 25 epoch, AdamW lr `1e-4`. Khi chuyển giai đoạn, optimizer và ReduceLROnPlateau **được khởi tạo lại** để trạng thái của giai đoạn trước không làm sai learning rate.
- Batch 32, weight decay `1e-4`, cross-entropy có class weights tính **chỉ từ train**; lớp `trash` có trọng số khoảng 3,064. Scheduler giảm lr ×0,5 khi validation macro-F1 không cải thiện sau patience 2. Early stopping patience 5 ở từng giai đoạn. Checkpoint `best.pt` chọn theo validation macro-F1 cao nhất toàn run; `last.pt` dùng resume.
- Mỗi epoch đánh giá **toàn bộ validation** 378 ảnh, lưu loss, accuracy, macro-F1 và confusion matrix/report của epoch tốt nhất. `test` không được nạp trong `src.train`; bộ 300 ảnh camera cuối cũng chưa dùng.
- Checkpoint lưu kiến trúc, weights ID, nhãn, preprocessing, config, manifest SHA-256, seed, phiên bản thư viện, model/optimizer/scheduler, RNG state và history. Inference tạo kiến trúc với `weights=None`, nên khi chạy checkpoint trên Mac không cần tải lại ImageNet.

## Bằng chứng kiểm thử kỹ thuật

- `runs/phase3_smoke_v2/`: chạy một train batch/epoch trên CPU, một epoch classifier, dừng có chủ đích, `--resume` rồi một epoch fine-tune. `last.pt` có đúng hai hàng history và trạng thái `complete`.
- Checkpoint smoke được đánh dấu `smoke: true`; macro-F1 validation 0,0533 là chỉ báo mã chạy được với **khởi tạo ngẫu nhiên và một batch**, không phải hiệu năng mô hình.
- `src.predict` nạp `best.pt` trên CPU và trả sáu xác suất có tổng bằng 1. Kiểm thử round-trip so sánh xác suất trước/sau lưu nạp tới 7 chữ số thập phân.
- 10/10 kiểm thử dữ liệu/mô hình qua cục bộ. Trọng số pretrained được nạp thành công, lớp cuối là `Linear(in_features=1024, out_features=6)`.

## Lượt huấn luyện thật `local_pretrained_probe`

Run được bắt đầu trên Mac MPS rồi tiếp tục từ `last.pt` sau epoch 1; cùng config, manifest và seed 42. Không giới hạn train batches, mỗi epoch đánh giá toàn bộ validation. Tổng **18 epoch**: 8 classifier + 10 fine-tune; giai đoạn fine-tune dừng sớm sau 5 epoch không cải thiện.

| Chỉ số của `best.pt` | Giá trị |
| --- | ---: |
| Epoch tốt nhất | 13 toàn run = fine-tune epoch 5 |
| Validation accuracy | 0,8730 (330/378 ảnh) |
| Validation macro-F1 | 0,8637 |
| Validation loss | 0,4075 |
| F1 `cardboard` / `glass` / `metal` | 0,932 / 0,872 / 0,900 |
| F1 `paper` / `plastic` / `trash` | 0,871 / 0,841 / 0,766 |

`best.pt`: `runs/local_pretrained_probe/best.pt`, SHA-256 `083945c828c8dbc7910413f8a15320a00e1425b8e7b8f39156becb0f71942d32`. Manifest SHA-256 `eea2338ba07941cd61abff4703734126b0275f4a51bad4d664d09ab4f63dd1de`. Bằng chứng: `run_summary.json`, `history.csv`, `history.json`, `learning_curves.png`, `best_validation_metrics.json` cùng thư mục. Các tệp `runs/` không đưa lên Git.

Một ảnh **validation** `glass/glass105.jpg` được nạp qua `src.predict` trên CPU: nhãn `glass`, softmax 0,9983; sáu điểm có trong đầu ra lệnh. Đây chỉ là ví dụ minh họa, **không phải** accuracy của camera và không chứng minh mô hình tự tin đúng trên dữ liệu ngoài miền. Bộ test TrashNet và camera final vẫn chưa được mở.

Đồ thị cho thấy train loss tiếp tục giảm tới 0,084 ở epoch 18 trong khi validation macro-F1 đạt đỉnh ở epoch 13 rồi không cải thiện; điều này gợi ý overfitting sau điểm tốt nhất. Lớp `trash` có F1 thấp nhất (0,766) và chỉ 21 ảnh validation nên chỉ số này có độ bất định đáng kể. Không dùng test để sửa riêng lỗi của lớp này; Phase 5 sẽ đánh giá mù sau khi chốt mô hình.

## Cách chạy

```bash
# Smoke test không tải trọng số; không dùng checkpoint này để báo cáo hiệu năng
uv run --locked python -m src.train --config configs/smoke.yaml --run-dir runs/another_smoke --smoke-batches 1 --no-pretrained

# Huấn luyện thật (Mac MPS đã dùng; Colab có notebook tái chạy)
uv run --locked python -m src.train --run-dir runs/baseline_v1

# Nếu phiên ngắt sau ít nhất một epoch
uv run --locked python -m src.train --run-dir runs/baseline_v1 --resume

# Dự đoán ảnh validation, không dùng ảnh test để chọn mô hình
uv run --locked python -m src.predict --checkpoint runs/baseline_v1/best.pt --image data/raw/dataset-resized/glass/glass105.jpg
```

Notebook `notebooks/01_huan_luyen_colab.ipynb` nhận `artifacts/phase3_colab_bundle.zip`, gọi cùng `src.train` và lưu kết quả trong Drive. Dữ liệu và mã nguồn trong gói được kiểm tra checksum. Notebook là phương án tái chạy trên T4, **chưa có đầu ra Colab cho Phase 3**. Lượt huấn luyện MPS ở trên là nguồn duy nhất của các chỉ số đã báo cáo.

## Đoạn nháp để bảo vệ

> MobileNetV3-Small được dùng làm backbone transfer learning. Đầu ra 1.000 lớp ImageNet được thay bằng đầu ra sáu lớp của TrashNet. Ban đầu chỉ cập nhật classifier để học ánh xạ đặc trưng sang nhãn rác mà không phá vỡ đặc trưng đã học; sau đó mở backbone với learning rate nhỏ hơn để thích nghi với miền ảnh TrashNet. Cross-entropy dùng trọng số nghịch tần suất từ train do lớp `trash` ít ảnh. Validation macro-F1 là tiêu chí chọn checkpoint/early stopping vì nó cho mỗi lớp trọng số ngang nhau; accuracy riêng lẻ có thể che giấu hiệu năng kém ở lớp ít mẫu. Test được giữ đến khi mô hình và ngưỡng đã khóa.

Logits là sáu giá trị chưa chuẩn hóa của mô hình; softmax đổi thành sáu xác suất cộng lại bằng 1. Điểm softmax cao không chứng minh độ tin cậy đã hiệu chỉnh. Nếu training loss giảm nhưng validation macro-F1 giảm hoặc đứng yên, đó là dấu hiệu overfitting/khác phân bố; không được lấy test để chỉnh siêu tham số.
