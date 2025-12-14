# Network Intrusion Detection System (NIDS) using Deep Learning

## 📌 Giới thiệu
Dự án nghiên cứu và phát triển hệ thống phát hiện xâm nhập mạng sử dụng các kỹ thuật Deep Learning tiên tiến (LSTM, CNN, Transformer). Hệ thống được thiết kế để xử lý dữ liệu lưu lượng mạng thực tế, phát hiện các cuộc tấn công và thích ứng với hiện tượng Concept Drift.

## 🚀 Tính năng chính
- **Đa mô hình:** Hỗ trợ LSTM, CNN-GRU, Attention và Transformer để so sánh hiệu năng.
- **Xử lý dữ liệu lớn:** Tích hợp pipeline xử lý cho các dataset CIC-IDS2017, CIC-DDoS2019, UNSW-NB15,...
- **Concept Drift:** Cơ chế phát hiện và cập nhật mô hình khi dữ liệu mạng thay đổi theo thời gian.
- **Giao diện trực quan:** (Nếu có Streamlit/Web) Hiển thị kết quả dự đoán thời gian thực.

## 🛠 Cài đặt
Yêu cầu: Python 3.11+

1. **Clone dự án:**
   ```bash
   git clone [https://github.com/NguyenPhuocAnhDung/DoAnChuyenNganh.git](https://github.com/NguyenPhuocAnhDung/DoAnChuyenNganh.git)
   cd DoAnChuyenNganh# Đề án Phát hiện Drift & Active Learning trên Dữ liệu Mạng

## Mô tả
Kho mã này triển khai pipeline xử lý, tiền xử lý, mô phỏng streaming, phát hiện drift, và cơ chế active learning cho dữ liệu mạng (CIC / CICEVSE). Bao gồm script chuẩn hóa dữ liệu nguồn, tạo stream, mô hình (CNN / Transformer SOTA), và các tiện ích vẽ báo cáo/so sánh.

## Cài đặt
1. Tạo môi trường ảo Python và cài dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # trên Linux/macOS
.venv\Scripts\activate      # trên Windows
pip install -r requirements.txt
```
2. (Tùy chọn) Khởi chạy stack bằng Docker Compose:
```bash
docker-compose up --build
```

## Sử dụng nhanh
- Script chính tổng quát: [src/system.py](src/system.py) — điểm vào hệ thống/chuẩn hóa luồng (mở file để xem hướng dẫn chi tiết).
- Chuẩn bị source data: gọi [`process_source_data`](src/setupstreaming/bodatamoi.py).
- Tách / xử lý dữ liệu drift: gọi [`process_and_merge_data`](src/setupstreaming/process_drift_data.py).
- Tự động tìm file attacker mẫu: gọi [`find_attacker_auto`](src/setupstreaming/prepare_data.py).
- Tiền xử lý streaming: xem [src/model/preprocess_stream.py](src/model/preprocess_stream.py) (hàm [`standardize_label`](src/model/preprocess_stream.py) được dùng để chuẩn hóa nhãn).
- Mô hình SOTA (Transformer): mạng encoder trong [src/model/main_sota.py](src/model/main_sota.py) — hàm [`transformer_encoder`](src/model/main_sota.py).
- Demo Kafka / real-time plotting: [src/demo_kafka/consumer_active.py](src/demo_kafka/consumer_active.py) và [src/demo_kafka/consumer_two.py](src/demo_kafka/consumer_two.py) (vẽ Accuracy vs Uncertainty, đánh dấu retraining).
- Script xuất báo cáo/biểu đồ: [src/model/plot_comparison.py](src/model/plot_comparison.py) và [src/model/xuata.py](src/model/xuata.py).

Chạy một ví dụ pipeline (tổng quan):
1. Chuẩn bị SOURCE: chạy [`process_source_data`](src/setupstreaming/bodatamoi.py).
2. Tạo dữ liệu drift / merge: chạy [`process_and_merge_data`](src/setupstreaming/process_drift_data.py).
3. Tiền xử lý & huấn luyện mô hình: xem các file trong [src/model/](src/model/).
4. Mô phỏng streaming / consumer: chạy các script trong [src/demo_kafka/](src/demo_kafka/).

## Cấu trúc dự án
- [README.md](README.md) — tài liệu này
- [requirements.txt](requirements.txt) — thư viện Python cần thiết
- [docker-compose.yml](docker-compose.yml) — container setup (nếu dùng)
- baocao/ — kết quả báo cáo, plots, báo cáo so sánh
  - baocao/main_cnn_attention/reports (ví dụ báo cáo drift)
- dataset/
  - processed/, raw/, processedstreamvs2.4/ — dữ liệu nguồn & đã xử lý
- results/
  - models/, plots/, comparison_plots/, final_comparison_plots/
- Sosanh/ — scripts/plots phục vụ so sánh
- src/ — mã nguồn chính
  - [src/system.py](src/system.py) — entry / cấu hình hệ thống
  - demo_kafka/
    - [src/demo_kafka/consumer_active.py](src/demo_kafka/consumer_active.py) — consumer realtime + active learning plotting
    - [src/demo_kafka/consumer_two.py](src/demo_kafka/consumer_two.py) — consumer/visualization variant
  - model/
    - [src/model/main_sota.py](src/model/main_sota.py) — cấu trúc Transformer (hàm [`transformer_encoder`](src/model/main_sota.py))
    - [src/model/main_cnn_attention.py](src/model/main_cnn_attention.py) — mô hình CNN + attention & plotting
    - [src/model/preprocess_stream.py](src/model/preprocess_stream.py) — tiền xử lý luồng (hàm [`standardize_label`](src/model/preprocess_stream.py))
    - [src/model/plot_comparison.py](src/model/plot_comparison.py) — tạo biểu đồ so sánh
    - [src/model/xuata.py](src/model/xuata.py) — script xuất báo cáo từ CSV drift
  - setupstreaming/
    - [src/setupstreaming/bodatamoi.py](src/setupstreaming/bodatamoi.py) — tạo SOURCE DATA (hàm [`process_source_data`](src/setupstreaming/bodatamoi.py))
    - [src/setupstreaming/prepare_data.py](src/setupstreaming/prepare_data.py) — chuẩn bị dữ liệu (hàm [`find_attacker_auto`](src/setupstreaming/prepare_data.py))
    - [src/setupstreaming/process_drift_data.py](src/setupstreaming/process_drift_data.py) — xử lý & merge dữ liệu drift (hàm [`process_and_merge_data`](src/setupstreaming/process_drift_data.py))
  - system_config.csv — cấu hình hệ thống mẫu

## Ghi chú ngắn
- Nhiều đường dẫn dữ liệu trong scripts là đường dẫn tuyệt đối (ví dụ trong `src/setupstreaming/`), hãy chỉnh lại theo môi trường của bạn.
- Kiểm tra kỹ các file trong `baocao/` và `results/` để có các file log/plot sẵn có.

Nếu cần hướng dẫn chạy từng script cụ thể, mở trực tiếp file tương ứng ở:
- [src/setupstreaming/bodatamoi.py](src/setupstreaming/bodatamoi.py)
- [src/setupstreaming/process_drift_data.py](src/setupstreaming/process_drift_data.py)
- [src/demo_kafka/consumer_active.py](src/demo_kafka/consumer_active.py)
