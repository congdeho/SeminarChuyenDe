# Quy trình chụp và gán nhãn ảnh camera

Tài liệu này chốt quy tắc **trước khi xem dự đoán**. Hiện chưa chụp bộ ảnh riêng; chỉ bắt đầu khi chuẩn bị đủ vật thể. Ảnh tự chụp không đưa vào train TrashNet ở Phase 3 trừ khi thiết kế lại thí nghiệm và khóa manifest mới.

## Hai bộ tách biệt

| Bộ | Mục đích | Số lượng | Vật thể |
| --- | --- | ---: | --- |
| `camera_dev` | Thử camera, kiểm tra ánh sáng/giao diện, chọn ngưỡng nếu cần | 60 = 10 ảnh/lớp | 2 vật thể/lớp × 5 ảnh; **không dùng lại trong final** |
| `camera_final` | Đánh giá ngoài miền ở Phase 5, mở một lần sau khi khóa mô hình | 300 = 50 ảnh/lớp | 10 vật thể/lớp × 5 ảnh; ít nhất 3 buổi chụp |

Tổng tối thiểu **12 vật thể khác nhau trên mỗi lớp** nếu theo đúng thiết kế này. Vật thể test không được là phiên bản xoay/đặt lại của vật thể dev. Mỗi ảnh chỉ có **một vật chính** chiếm phần lớn khung hình; không để nhiều loại rác cùng xuất hiện. Không dùng ảnh chụp từ màn hình, ảnh tải mạng hoặc ảnh TrashNet.

## Định nghĩa nhãn thực hành

- `cardboard`: hộp/tấm bìa carton là vật liệu chính.
- `glass`: chai, lọ hoặc vật thủy tinh là vật liệu chính.
- `metal`: lon, hộp hoặc vật kim loại là vật liệu chính.
- `paper`: giấy rời, báo, tờ in là vật liệu chính; không coi carton là paper.
- `plastic`: chai, hộp hoặc đồ nhựa cứng là vật liệu chính.
- `trash`: vật không khớp năm lớp trên, ví dụ xốp, khăn giấy đã dùng sạch về mặt an toàn, bao bì nhiều lớp không tách được; ghi mô tả cụ thể.

Không cố ép nhãn cho vật có hai chất liệu tương đương, đồ điện tử, rác nguy hại, chất thải hữu cơ, chất lỏng, hoặc ảnh không rõ vật chính. Đưa các trường hợp này vào danh sách `out_of_scope`, **không lén thay bằng `trash`**. Đây là phân loại hình ảnh theo sáu nhãn của dữ liệu, không phải hướng dẫn xử lý/tái chế theo quy định địa phương.

## Cách chụp

1. Gán `object_id` duy nhất cho **vật thể vật lý**, ví dụ `glass_final_03`. Ghi `session_id`, ngày/giờ, nhãn trước khi chạy mô hình. Cùng vật dù chụp nhiều lần phải giữ nguyên `object_id`.
2. Mỗi vật chụp 5 ảnh: chính diện, xoay trái, xoay phải, gần hơn, xa hơn/đổi ánh sáng. Trộn ít nhất 3 nền/điều kiện trong toàn bộ bộ final (bàn sáng, nền tối, ánh sáng cửa sổ/đèn). Không làm tối đến mức người cũng không nhận ra.
3. Dùng camera sẽ trình diễn, lưu ảnh gốc JPG/PNG, không crop hoặc lọc hậu kỳ. Để vật ở giữa nhưng không luôn cùng vị trí; giữ camera ổn định, tránh nhòe và che khuất vật chính.
4. Chụp qua tối thiểu 3 buổi khác nhau; ghi `session_id` để có thể phân tích phụ thuộc theo buổi. Trộn thứ tự các lớp giữa các buổi, không chụp toàn bộ một lớp trong một bối cảnh duy nhất.
5. Tự gán nhãn lần đầu khi chụp. Sau ít nhất một lượt nghỉ, kiểm tra nhãn lần hai **trước khi xem dự đoán**; ghi thay đổi và lý do. Đây là tự kiểm tra hai lượt, **không phải hai người gán nhãn độc lập**.
6. Loại ảnh lỗi kỹ thuật theo tiêu chí định trước (không thấy vật chính, file hỏng, mờ không nhận diện được) và ghi lý do trước khi chạy mô hình. Không loại ảnh chỉ vì mô hình đoán sai.

## Mẫu metadata

Lưu CSV cạnh ảnh với các cột:

`path,label,object_id,split,session_id,captured_at,background,lighting,label_reviewed,review_note,exclude_reason`

`split` chỉ nhận `camera_dev` hoặc `camera_final`. Ví dụ: `camera_final/glass_final_03_02.jpg,glass,glass_final_03,camera_final,S2,2026-10-10T14:00:00+07:00,ban_go,den_trong_phong,yes,,`.

Trước khi đánh giá, kiểm tra: mỗi lớp đúng số ảnh/vật thể, không trùng `object_id` giữa dev/final, không trùng SHA-256 giữa hai bộ, metadata không thiếu và mọi ảnh mở được. 300 ảnh là 300 **lần chụp**, chỉ có 60 vật thể độc lập; không diễn giải khoảng tin cậy như 300 quan sát độc lập. Có thể báo cáo thêm độ chính xác theo vật thể/buổi.

## Danh sách chuẩn bị 12 vật thể/lớp (gợi ý)

Tìm các vật thể **khác nhau thật**, không chỉ 12 góc của một món: 12 hộp/tấm carton; 12 chai/lọ thủy tinh; 12 lon/hộp kim loại; 12 tờ/bó giấy/báo khác nhau; 12 chai/hộp nhựa; 12 món rác khác thuộc định nghĩa trên. Đánh số 01–02 cho dev, 03–12 cho final. Vệ sinh và ưu tiên vật an toàn; không xử lý rác nguy hại chỉ để đủ mẫu.
