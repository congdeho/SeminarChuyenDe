# Máy trình diễn và môi trường

Kiểm tra trên máy cục bộ ngày 06/10/2026:

| Mục | Kết quả quan sát |
|---|---|
| Máy | MacBook Air, model Mac16,12 |
| Chip | Apple M4, 10 lõi CPU |
| RAM | 16 GB |
| Hệ điều hành | macOS 15.7.4, arm64 |
| Python hệ thống | 3.14.5 |
| `uv` | 0.12.13 |
| Git | 2.50.1 |
| Thư viện học máy trong Python hệ thống | Chưa có tại lúc kiểm tra |

Kiểm tra môi trường dự án trong cùng ngày:

| Mục | Kết quả thực tế |
|---|---|
| Python dự án | 3.12.14 trong `.venv` |
| `torch` / `torchvision` | 2.14.1 / 0.29.1 |
| `opencv-python` / `numpy` | 4.14.0.94 / 2.5.3 |
| `scikit-learn` / `matplotlib` | 1.9.1 / 3.11.2 |
| `pillow` / `pandas` | 12.3.0 / 3.0.6 |
| `pyyaml` / `tqdm` | 6.0.3 / 4.70.1 |
| PyTorch CPU | Hoạt động; kiểm tra tensor nhỏ thành công |
| MPS | Không khả dụng trong sandbox; **khả dụng khi chạy ngoài sandbox** và đã hoàn thành 18 epoch Phase 3 trên Mac MPS. Video giả lập Phase 4 cũng đã chạy MPS. |
| CUDA cục bộ | Không khả dụng trên Mac này |
| Khóa thư viện | `uv.lock` được tạo từ `pyproject.toml` và `uv sync --locked` cài thành công |

Kiểm tra Colab trên tài khoản người thực hiện cùng ngày bằng notebook [Seminar - kiem tra GPU va thu vien](https://colab.research.google.com/drive/18GLYCrcuz-SKP6hszscWJF81S2-Eh59q):

| Mục | Kết quả thực tế |
|---|---|
| Loại runtime | GPU T4 |
| Python | 3.13.15 |
| PyTorch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| NumPy / Pillow | 2.1.3 / 11.3.0 |
| OpenCV / scikit-learn | 5.0.0.93 / 1.6.1 |
| CUDA | `True`; GPU `Tesla T4` |
| Tensor smoke test | `cuda:0`; tổng của tensor 2 × 2 bằng 4.0 |

Colab đang dùng Python 3.13, khác Python cục bộ 3.12. Hai môi trường cũng khác phiên bản PyTorch, torchvision và OpenCV. Notebook huấn luyện Phase 3 đã tạo nhưng **chưa chạy trên Colab**; kết quả mô hình hiện tại chỉ từ Mac MPS. Nếu tái chạy Colab, ghi phiên bản runtime và kiểm thử mã chung; không cài gói Mac từ `uv.lock` trực tiếp vào Colab Linux/GPU.

`sysctl` không trả được RAM/CPU trong sandbox; thông tin phần cứng ở bảng trên được lấy bằng `system_profiler SPHardwareDataType`. Không ghi số serial hoặc định danh cá nhân vào tài liệu.

## Lưu kết quả kiểm tra sau cài đặt

Chạy `uv run --locked python -m src.check_env` tại thư mục dự án. Phiên bản cục bộ trên đã được lấy từ đầu ra lệnh này và `uv sync`; chạy lại khi môi trường thay đổi. Với Colab, chạy lại notebook mỗi khi mở runtime mới hoặc trước thí nghiệm chính. Không tự chép chỉ số phần cứng hoặc phiên bản từ tài liệu trên mạng.
