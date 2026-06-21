# 🔬 Hướng Dẫn Sử Dụng & Mô Tả Chi Tiết Ứng Dụng Web AMR Analyzer
## Hệ Thống Hỗ Trợ Quyết Định Lâm Sàng & Giải Thích Kháng Thuốc Ciprofloxacin ở Vi Khuẩn *Escherichia coli*

Ứng dụng Web **AMR Analyzer** là giao diện tương tác trực quan của mô hình học máy (Machine Learning Pipeline) dự đoán tính kháng thuốc **Ciprofloxacin** của vi khuẩn *E. coli*. Giao diện được thiết kế tối ưu dành cho các nghiên cứu viên sinh tin học và bác sĩ vi sinh lâm sàng, giúp chuyển đổi các chỉ số xác suất khô khan thành các đề xuất điều trị y khoa có bằng chứng khoa học và giải thích cơ chế rõ ràng.

---

## 🎨 1. Thiết Kế Giao Diện & Trải Nghiệm Người Dũng (UX/UI)

Hệ thống được xây dựng trên ngôn ngữ thiết kế **Premium Glassmorphism** hiện đại kết hợp Sleek Dark Mode, mang lại trải nghiệm làm việc chuyên nghiệp, giảm mỏi mắt khi sử dụng liên tục trong phòng thí nghiệm:
* **Hệ màu sắc tương phản cao (High-Contrast System):** 
  * Nền tối sâu thẳm phối hợp với các dải chuyển màu (gradient) xanh tím thời thượng.
  * Chỉ thị trạng thái rõ ràng: Màu đỏ Tomato đại diện cho trạng thái **Kháng thuốc (Resistant)**; Màu xanh lục Emerald đại diện cho trạng thái **Nhạy cảm (Susceptible)**; Màu vàng hổ phách cảnh báo đối tượng đặc biệt.
* **Biểu đồ động tương tác:** Tích hợp thư viện **Chart.js** để vẽ các biểu đồ tròn (gauge) đo lường xác suất, biểu đồ thanh ngang SHAP và biểu đồ đường dịch tễ học sắc nét, mượt mà khi di chuột qua.
* **Micro-animations:** Các hiệu ứng chuyển động nhỏ khi di chuột (hover), hiệu ứng loading spinner xoay tròn khi tính toán SHAP/AI giúp giao diện phản hồi sinh động, không gây cảm giác chờ đợi thụ động.

---

## 🛠️ 2. Các Phân Hệ Tính Năng Cốt Lõi

### Phân hệ 1: Dashboard Tổng Quan (Model Evaluation & Performance)
* **Thông số mô hình:** Hiển thị tên mô hình đang chạy thực tế, tổng số đặc trưng (Features) đầu vào và ngưỡng quyết định tối ưu lâm sàng (Threshold).
* **Chỉ số kiểm thử:** Trình bày trực quan 3 biểu đồ tròn đo lường chất lượng mô hình trên tập kiểm thử độc lập: **Accuracy (83.00%)**, **ROC-AUC (90.29%)**, và **PR-AUC (89.05%)**.
* **Bảng so sánh phương pháp:** Đối chiếu hiệu năng giữa các thuật toán (Baseline, Stacking Ensemble, XGBoost Pipeline) giúp nghiên cứu viên đánh giá tính khoa học của hệ thống học máy.

### Phân hệ 2: Chẩn Đoán Đơn Chủng (Single Diagnostic)
Đây là tính năng cốt lõi nhất phục vụ trực tiếp cho bác sĩ lâm sàng khi phân tích một mẫu bệnh phẩm cụ thể:
* **Nhập ma trận đặc trưng:** Hỗ trợ nhập trực tiếp dữ liệu biểu hiện gen và k-mer dưới dạng văn bản JSON.
* **Nạp mẫu ngẫu nhiên (Random Sampling):** Hai nút **"Nạp mẫu nhạy cảm"** và **"Nạp mẫu kháng thuốc"** sẽ tự động gọi API lên máy chủ để lấy ngẫu nhiên các kiểu gen bệnh nhân khác nhau từ tệp dữ liệu kiểm thử, giúp chạy thử nghiệm nhanh chóng.
* **Thước đo xác suất đề kháng (AMR Gauge):** Thể hiện trực quan mức độ rủi ro kháng thuốc của vi khuẩn theo thang đo phần trăm. Nếu vượt quá ngưỡng quyết định (0.521), hệ thống sẽ cảnh báo trạng thái **Kháng thuốc (Resistant)** màu đỏ đậm.
* **Đồ thị giải thích SHAP cục bộ (Explainable AI - XAI):**
  * Vẽ biểu đồ thanh ngang định lượng mức độ đóng góp của từng gen/k-mer cụ thể vào quyết định của mô hình.
  * **Cột Đỏ (SHAP dương):** Các yếu tố sinh học thúc đẩy chủng vi khuẩn trở nên kháng thuốc (ví dụ: đột biến đích QRDR `gyrA_S83L`, `parC_S80I` hoặc gen sinh enzyme ESBL `blaCTX-M-15`).
  * **Cột Xanh (SHAP âm):** Các yếu tố kiểu gen giữ chủng vi khuẩn ở trạng thái nhạy cảm.
* **Báo cáo chuyên khoa AI lâm sàng (AI Clinical Report):**
  * Tích hợp **CARD Database (Từ điển gen):** Tự động truy vấn và giải nghĩa tiếng Việt chi tiết cho hơn 200 gen kháng thuốc phát hiện trong mẫu.
  * Tích hợp **RAG PDF Engine (Retrieval-Augmented Generation):** Sử dụng trí tuệ nhân tạo (Gemini API hoặc Hệ chuyên gia cục bộ) để đọc và trích xuất thông tin điều trị từ tài liệu chính thức: *"Hướng dẫn sử dụng kháng sinh"* của Bộ Y tế Việt Nam. Báo cáo tự động in rõ số trang và tên tài liệu tham chiếu (Ví dụ: *"Theo Hướng dẫn sử dụng kháng sinh của Bộ Y tế, Trang 23..."*) giúp bác sĩ yên tâm đối chiếu.
* **Trợ lý Lâm sàng AI (Clinical AI Chatbot):** Cung cấp cổng chat trực tiếp tại giao diện. Bác sĩ có thể đặt câu hỏi tự do cho AI về ca bệnh (Ví dụ: *"Chủng vi khuẩn mang đột biến gyrA này kê đơn Ciprofloxacin được không?"*, *"Bệnh nhân mang thai và dị ứng penicillin thì dùng thuốc thay thế gì?"*). Trợ lý sẽ trả lời dựa trên cuốn sách PDF y khoa và lưu ý chống chỉ định cụ thể theo độ tuổi/thai kỳ.

### Phân hệ 3: Chẩn Đoán Hàng Loạt (Batch Diagnostics)
* **Kéo thả tệp dữ liệu:** Cho phép người dùng kéo thả file `.csv` chứa ma trận đặc trưng của hàng trăm, hàng ngàn chủng vi khuẩn cần phân tích đồng thời.
* **Xem trước kết quả (Inference Preview):** Hiển thị bảng kết quả dự đoán của 50 mẫu đầu tiên (bao gồm mã chủng, kết luận nhãn, và xác suất kháng thuốc cụ thể).
* **Tải báo cáo tổng hợp:** Nút **"Tải kết quả đầy đủ (CSV)"** cho phép tải xuống tệp dữ liệu đã được mô hình dán nhãn dự đoán đầy đủ phục vụ cho các phân tích thống kê quy mô lớn.

### Phân hệ 4: Lịch Sử Chẩn Đoán (History Logs)
* **Cơ sở dữ liệu SQLite cục bộ:** Tự động lưu lại tất cả các ca bệnh đã chẩn đoán kèm theo mã bệnh nhân, ngày giờ thực tế, kết luận chẩn đoán, xác suất và danh sách gen phát hiện.
* **Xem lại chi tiết:** Khi click vào một hàng bất kỳ trong bảng lịch sử, hệ thống sẽ tự động tải lại toàn bộ đồ thị biểu đồ SHAP và báo cáo AI của ca bệnh đó mà không làm mất thời gian tính toán lại từ đầu.
* **Quản lý lịch sử:** Hỗ trợ xóa từng bản ghi hoặc xóa sạch toàn bộ lịch sử để giải phóng dung lượng.

### Phân hệ 5: Giám Sát Dịch Tễ Học (Epidemiological Surveillance)
* **Xu hướng kháng thuốc theo thời gian:** Vẽ biểu đồ đường thể hiện tỷ lệ phần trăm mẫu kháng thuốc thay đổi theo các ngày trong tháng, giúp phát hiện sớm các ổ dịch hoặc đợt bùng phát vi khuẩn đa kháng thuốc.
* **Biểu đồ phân bố tần suất gen:** Thống kê các gen kháng thuốc điển hình xuất hiện nhiều nhất trong số các ca bệnh đã chẩn đoán, giúp quản lý y tế đưa ra chính sách kiểm soát kháng sinh phù hợp tại bệnh viện.

---

## 🔬 3. Hướng Dẫn Vận Hành Cho Người Dùng

Để sử dụng ứng dụng web một cách tối ưu, hãy làm theo quy trình đề xuất dưới đây:

### Bước 1: Khởi động Server
Mở Terminal tại thư mục dự án và chạy lệnh:
```bash
python run_web_app.py
```
Sau đó mở trình duyệt web và truy cập địa chỉ: `http://127.0.0.1:5000`.

### Bước 2: Nạp dữ liệu ca bệnh
* **Cách A (Test nhanh):** Nhấn nút **"Nạp mẫu kháng thuốc"** hoặc **"Nạp mẫu nhạy cảm"** ở góc trái. Hệ thống sẽ tự động điền ngẫu nhiên dữ liệu đặc trưng vào ô nhập JSON.
* **Cách B (Chẩn đoán thực tế):** Nhập mã bệnh nhân vào ô *"Mã bệnh nhân"* và dán chuỗi JSON đặc trưng kiểu gen thu được sau giải trình tự vào ô nhập liệu.

### Bước 3: Chạy chẩn đoán và đọc biểu đồ
* Nhấn nút **"Dự đoán & Giải thích"**. 
* Quan sát biểu đồ tròn đo xác suất và biểu đồ SHAP ở giữa màn hình để xác định những gen nào đang đóng vai trò chính đẩy vi khuẩn vào trạng thái kháng thuốc.

### Bước 4: Tham chiếu báo cáo và tư vấn AI
* Đọc báo cáo lâm sàng được sinh ra ở góc dưới. Đối chiếu các đoạn trích dẫn của Bộ Y tế (nếu có) được trích ra dưới cùng của báo cáo.
* Sử dụng khung chat ở bên phải, nhập câu hỏi của bạn để trò chuyện trực tiếp với Trợ lý AI về ca bệnh phức tạp này.

### Bước 5: Xem lại lịch sử hoặc Giám sát dịch tễ
* Chuyển sang tab **"Lịch sử chẩn đoán"** ở thanh điều hướng phía trên để xem lại các ca bệnh cũ.
* Chuyển sang tab **"Thống kê dịch tễ học"** để theo dõi biểu đồ xu hướng kháng thuốc tổng thể của bệnh viện.
