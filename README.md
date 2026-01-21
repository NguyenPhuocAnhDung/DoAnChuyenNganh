# CD-AHAL: Concept Drift Aware Hybrid Active Learning for NIDS

## 📖 Giới thiệu
Dự án xây dựng hệ thống phát hiện xâm nhập mạng (NIDS) sử dụng Deep Learning, tập trung vào khả năng xử lý dữ liệu luồng (Streaming Data) và thích nghi với Concept Drift. Hệ thống tích hợp pipeline từ tiền xử lý dữ liệu, training offline, đến giả lập môi trường Streaming với Kafka.

## Dataset
Link: ```(https://drive.google.com/file/d/1oFByJODMDo3jw5POQUGjMULs-MJVhNhq/view?usp=drive_link)```

## 🏗️ Cấu trúc dự án
Dự án được tổ chức theo cấu trúc module hóa:

* **`src/`**: Mã nguồn chính.
    * `model/`: Chứa các kịch bản huấn luyện (`main_cnn_attention.py`, `main_cnn_gru_attention.py`, `main_active_learning.py`...).
    * `setupstreaming/`: Các module xử lý dữ liệu luồng và chuẩn bị dữ liệu (`process_drift_data.py`).
    * `demo_kafka/`: Giả lập hệ thống Real-time với Producer/Consumer (`consumer.py`, `producer.py`).
    * `sosanhchiso/`: Scripts vẽ biểu đồ và so sánh hiệu năng các model.
* **`dataset/`**:
    * `processed_v3_unified/`: Dữ liệu đã tiền xử lý dạng Parquet.
    * `raw/`: Dữ liệu thô (CIC-IDS, UNSW-NB15...).
* **`baocao/` & `results/`**: Lưu trữ biểu đồ (Plots), Confusion Matrix và báo cáo kết quả so sánh.

## 🚀 Tính năng nổi bật
1.  **Đa dạng Model:** Hỗ trợ CNN-Attention, CNN-GRU, Generative Models và Active Learning.
2.  **Streaming Simulation:** Giả lập luồng dữ liệu mạng thực tế sử dụng Kafka.
3.  **Concept Drift Detection:** Phát hiện sự thay đổi phân phối dữ liệu mạng theo thời gian.
4.  **Explainable AI (XAI):** Tích hợp phân tích khả năng giải thích của model (như trong folder `plots/XAI_Batch...`).

## 🛠️ Cài đặt & Chạy thử
1.  **Môi trường:**
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Huấn luyện Model (Ví dụ CNN-Attention):**
    ```bash
    python src/model/main_cnn_attention.py
    ```

3.  **Chạy Streaming Demo:**
    Cần cài đặt Kafka và Docker (sử dụng `docker-compose.yml` có sẵn).