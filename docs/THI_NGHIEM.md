# Sổ thí nghiệm

Đã có smoke test kỹ thuật và một lượt huấn luyện đầy đủ trên Mac MPS. Chỉ điền số liệu từ các đầu ra chạy thật; không so macro-F1 smoke với mô hình đã huấn luyện. Mọi số dưới đây là **validation**, chưa phải test.

| Mã chạy | Ngày | Cấu hình | Seed | Manifest SHA-256 | Checkpoint | Validation macro-F1 | Ghi chú |
|---|---|---|---:|---|---|---:|---|
| `phase3_smoke_v2` | 06/10/2026 | `configs/smoke.yaml`; 1 batch/epoch; random init | 42 | `eea2338b…` | `runs/phase3_smoke_v2/best.pt` | 0,0533 | Chỉ kiểm thử lưu/resume/dự đoán; **không phải kết quả seminar** |
| `local_pretrained_probe` | 06/10/2026 | `configs/train.yaml`; 8 classifier + 10 fine-tune; MPS | 42 | `eea2338b…` | `runs/local_pretrained_probe/best.pt` (`083945c8…`) | **0,8637** | Best epoch 13; validation accuracy 0,8730; early stopping epoch 18; test chưa dùng |
| `phase5/e1_seed42` | 18/11/2026 | `configs/phase5_e1.yaml`; CNN từ đầu, 40 epoch; MPS | 42 | `eea2338b…` | `runs/phase5/e1_seed42/best.pt` (`865c2d9f…`) | **0,7031** | Best epoch 40; validation accuracy 0,7222; evaluator xác nhận; test chưa dùng |
| `phase5/e2_seed42` | 18/11/2026 | `configs/phase5_e2.yaml`; ResNet18 pretrained, 8+9 epoch; MPS | 42 | `eea2338b…` | `runs/phase5/e2_seed42/best.pt` (`611f9b90…`) | **0,8681** | Best epoch 12; validation accuracy 0,8783; evaluator xác nhận; test chưa dùng |
| `phase5/e4_seed42` | 18/11/2026 | `configs/phase5_e4.yaml`; MobileNet không augmentation, 8+14 epoch; MPS | 42 | `eea2338b…` | `runs/phase5/e4_seed42/best.pt` (`533cc4c8…`) | **0,8403** | Best epoch 17; validation accuracy 0,8519; evaluator xác nhận; test chưa dùng |

## Mẫu ghi sau mỗi lần chạy

```text
Mã chạy:
Ngày, máy, CPU/GPU:
Phiên bản Python/PyTorch/torchvision:
Lệnh và cấu hình:
Seed và checksum manifest:
Thời gian, epoch kết thúc:
Checkpoint và checksum:
Kết quả validation:
Lỗi/cảnh báo:
Kết luận và bước tiếp:
```
