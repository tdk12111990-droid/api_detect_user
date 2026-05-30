# 🌟 Hệ thống Camera Biên Thông minh (Smart Camera Edge System)

Hệ thống tích hợp đa luồng hiệu năng cao chạy trên **Jetson Orin Nano**, kết hợp bộ phát hiện người chuyên dụng bằng **TensorRT** và dịch vụ dự báo chuỗi thời gian nhiệt độ tự động tối ưu **FastAPI & TensorFlow (MTKAN)**.

---

## 🛠️ Kiến trúc hệ thống (System Architecture)

Dự án bao gồm 2 dịch vụ chính chạy song song dưới sự điều phối của một **Master Daemon Control** (`main_all.py`):

1. **YOLOv8n TensorRT Detector (Bộ Phát hiện Người)**:
   - Tối ưu hóa GPU biên thông qua việc biên dịch file ONNX sang TensorRT Engine (`yolov8n.engine`) chạy ở tốc độ cực cao (~251 FPS).
   - Thiết kế dạng chạy ngầm không màn hình (Headless Daemon), bắt giữ các luồng video RTSP bất đồng bộ và đẩy cảnh báo qua Thread Pool lên máy chủ trung tâm.
2. **FastAPI Thermal Service (Dịch vụ Nhận dữ liệu & Dự báo Nhiệt)**:
   - Nhận dữ liệu nhiệt độ trực tiếp qua API, lưu lịch sử tối ưu và đưa ra dự đoán trước 5 phút (Horizon = 5) bằng mạng Nơ-ron.
   - **Tự động huấn luyện lại ngầm (Background Auto-Retraining)**: Sau mỗi 2 ngày (172.800 giây), dịch vụ tự động kích hoạt train lại mô hình trên luồng phụ độc lập.
   - **Cập nhật nóng không gián đoạn (Zero-Downtime Hot-Swapping)**: Tự động nạp (hot-swap) trọng số mô hình mới vào RAM ngay sau khi train xong mà không cần khởi động lại server.

---

## 📋 Yêu cầu hệ thống (Prerequisites)

* **Thiết bị**: Jetson Orin Nano (hoặc dòng Jetson tương đương).
* **Hệ điều hành**: JetPack (5.x hoặc 6.x) đã tích hợp sẵn **CUDA**, **cuDNN**, và **TensorRT**.
* **Python**: Phiên bản 3.10.

---

## 🚀 Hướng dẫn cài đặt & Chạy dự án (Quick Start)

### Bước 1: Clone dự án và truy cập thư mục gốc
```bash
git clone <url-cua-ban> smart_camera_edge_system
cd smart_camera_edge_system
```

### Bước 2: Cấu hình Môi trường ảo cho Dịch vụ Thermal (TensorFlow)
Dịch vụ thermal chạy trong môi trường ảo chuyên dụng `tf_env` đặt trong thư mục `timeseries_service` để đảm bảo không bị xung đột thư viện:

```bash
cd timeseries_service
# Tạo môi trường ảo
python3 -m venv tf_env
# Kích hoạt môi trường ảo
source tf_env/bin/activate

# Nâng cấp pip và cài đặt thư viện
pip install --upgrade pip
pip install -r ../requirements.txt

# Cài đặt TensorFlow tương thích với Jetson (Nên dùng bản do NVIDIA cung cấp)
# Hoặc cài đặt bản chuẩn từ pip nếu chạy trên PC thông thường:
pip install tensorflow

deactivate
cd ..
```

### Bước 3: Cài đặt thư viện cho Bộ phát hiện người (YOLO TensorRT)
Ở thư mục gốc của dự án, cài đặt các thư viện phục vụ xử lý luồng camera và TensorRT:

```bash
# Cài đặt các thư viện cơ bản
pip install opencv-python requests pycuda
```

---

## ⚙️ Cấu hình Hệ thống (Configuration)

### 1. Cấu hình Camera RTSP & API Endpoint
Chỉnh sửa file `api_send.py` hoặc `yolo_detect.py` để thay đổi camera nguồn và URL đích:
* **`SOURCE`**: Đường dẫn RTSP của camera IP của bạn.
* **`API_URL`**: Địa chỉ server đích nhận cảnh báo phát hiện người.
* **`CONF_THRES`**: Ngưỡng tự tin phát hiện người (Mặc định: 0.55).

### 2. Cấu hình các điểm nhiệt độ giám sát
Chỉnh sửa file cấu hình tại `timeseries_service/model/config.json` để thay đổi danh sách các điểm nhiệt độ camera đo được:
* **`targets`**: Danh sách ID các điểm nhiệt (Mặc định: `ID_1` đến `ID_6`).

---

## 🎯 Chạy ứng dụng (Execution)

### Cách 1: Chạy song song cả hai dịch vụ bằng Master Daemon (Khuyên dùng)
Tại thư mục gốc của dự án, chạy lệnh:
```bash
python3 main_all.py
```
*Lệnh này sẽ tự động khởi động đồng thời cả camera giám sát phát hiện người và máy chủ API Thermal, đồng thời điều phối luồng log in ra màn hình vô cùng trực quan.*

### Cách 2: Chạy riêng lẻ từng dịch vụ để Debug

* **Chạy dịch vụ phát hiện người YOLO (TensorRT)**:
  ```bash
  python3 api_send.py
  ```

* **Chạy dịch vụ API Thermal & Dự báo**:
  ```bash
  cd timeseries_service
  ./tf_env/bin/python thermal_receiver.py
  ```

---

## ⚠️ Lưu ý quan trọng khi thay đổi điểm nhiệt độ

Khi người dùng **thêm điểm nhiệt mới** hoặc **đổi tên/vị trí điểm nhiệt** trên camera:
1. **Cập nhật Cấu hình**: Truy cập file `timeseries_service/model/config.json` và cập nhật lại danh sách điểm nhiệt mới trong mục `"targets"`.
2. **Huấn luyện lại**: Do kích thước ma trận đầu vào của mô hình AI thay đổi, bạn cần thực hiện huấn luyện lại mô hình từ đầu bằng lệnh:
   ```bash
   cd timeseries_service
   ./tf_env/bin/python train.py
   ```
   *Sau khi chương trình train xong và tạo ra trọng số mới, dịch vụ sẽ tự động nạp nóng và hoạt động ổn định bình thường.*
