**Quan sát 12 ảnh test có điểm sai cao nhất của ResNet18 seed 42.** Mô tả dưới đây dựa trên đặc điểm nhìn thấy trong ảnh. Chúng gợi ý các yếu tố có thể gây nhầm giữa lớp, chưa xác định mô hình thực sự dựa vào vùng ảnh nào.

| Ảnh | Quan sát trực tiếp |
|---|---|
| `plastic469.jpg` | Chai xanh xuyên sáng chụp rất gần từ hướng đáy. Thành chai cong và phản sáng giống bề mặt thủy tinh. |
| `trash104.jpg` | Mảnh rác màu nâu nhăn có bề mặt mờ, nếp gập và vân gần với giấy gói. |
| `glass102.jpg` | Vật trong mờ có phần mép mỏng và nhãn bong. Ánh sáng làm cụm này trông như bao bì nhựa. |
| `metal318.jpg` | Góc chụp từ trên cho thấy phần kim loại tròn, viền đỏ và phản quang mạnh. Mặt bên và chiều dày gần như khuất. |
| `plastic364.jpg` | Nhãn giấy có chữ và mã vạch chiếm phần lớn khung hình. Phần hộp nhựa hiện chủ yếu ở mép. |
| `plastic449.jpg` | Mảnh nhựa đen dài, nhăn và bắt sáng. Hình dạng ban đầu của bao bì khó nhận ra. |
| `plastic171.jpg` | Chai xanh trong, thành có gân và nắp bật. Hình dáng chai và độ trong có thể gợi lớp thủy tinh. |
| `paper171.jpg` | Mặt vật trắng mịn, có cạnh cong và vùng sáng phản chiếu. Dấu hiệu sợi giấy khó thấy. |
| `plastic303.jpg` | Hộp nhựa trong bị cắt ở biên dưới. Phần hiện rõ nhất là thành trong và viền nắp sẫm. |
| `plastic84.jpg` | Ảnh cận hộp có viền sáng và nhãn in màu. Một phần vật nằm ngoài khung hình. |
| `plastic452.jpg` | Mảnh nhựa đen nhỏ, nhăn và cách xa camera. Nền sáng chiếm phần lớn ảnh. |
| `trash52.jpg` | Bao bì dạng dải dẹt có nhiều chữ in, bề mặt phản sáng và mép xé. Hình dạng phẳng giống giấy in từ góc chụp này. |

Cả 12 dự đoán trên đều vượt ngưỡng 0,60 đã khóa. Trên toàn bộ test, cặp nhầm nhiều nhất là nhựa thành thủy tinh với 9 ảnh. Lớp `trash` có recall 75,00% sau khi nhận đúng 15/20 ảnh. Mỗi ảnh chiếm 5 điểm phần trăm recall của riêng lớp này.
