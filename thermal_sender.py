import time
import os
import json
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime

# --- CONFIGURATION ---
CAMERA_IP = "192.168.10.152"
CAMERA_USER = "admin"
CAMERA_PASS = "Demo@2024"

# Endpoint to fetch rules temperature from camera (Default Mode)
CAMERA_URL = f"http://{CAMERA_IP}/ISAPI/Thermal/channels/2/thermometry/1/rulesTemperatureInfo?format=json"

# Endpoint to fetch thermal matrix & image (P2P Mode)
P2P_URL = f"http://{CAMERA_IP}/ISAPI/Thermal/channels/2/thermometry/jpegPicWithAppendData?format=json"

# Server endpoint to send thermal data (Central Server)
SERVER_URL = "http://192.168.10.102:8100/api/thermal-data"

# Local receiver endpoint to trigger predictions
LOCAL_RECEIVER_URL = "http://127.0.0.1:8080/api/thermal-data"

# Path to custom points configuration saved by Frontend
POINTS_CONFIG_PATH = os.path.join("timeseries_service", "data", "thermal_points.json")

# Polling interval in seconds (5 minutes)
POLL_INTERVAL = 300

def parse_multipart(data, boundary):
    """Phân tách gói tin multipart từ Hikvision để lấy JSON metadata và P2P binary data."""
    parts = data.split(boundary)
    result = {}
    for part in parts:
        if b'Content-Type: application/json' in part:
            header_end = part.find(b'\r\n\r\n')
            result['json'] = json.loads(part[header_end+4:].strip())
        elif b'Content-Type: image/pjpeg' in part or b'Content-Type: image/jpeg' in part:
            header_end = part.find(b'\r\n\r\n')
            if 'image' not in result:
                result['image'] = part[header_end+4:]
        elif b'Content-Type: application/octet-stream' in part or len(part) > 100000:
            header_end = part.find(b'\r\n\r\n')
            if header_end != -1:
                result['p2p'] = part[header_end+4:]
    return result

def get_camera_temperatures():
    """Fetch temperatures. Prioritizes P2P coordinates from frontend config, falls back to camera rules."""
    auth = HTTPDigestAuth(CAMERA_USER, CAMERA_PASS)
    
    # 1. KIỂM TRA VÀ SỬ DỤNG CẤU HÌNH TỌA ĐỘ TỪ FRONTEND
    if os.path.exists(POINTS_CONFIG_PATH):
        try:
            with open(POINTS_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            custom_points = cfg.get("points", [])
            if custom_points:
                response = requests.get(P2P_URL, auth=auth, timeout=5)
                if response.status_code == 200:
                    content_type = response.headers.get("Content-Type", "")
                    boundary = b'--boundary'
                    if "boundary=" in content_type:
                        b_str = content_type.split("boundary=")[-1].strip()
                        boundary = f"--{b_str}".encode('utf-8')
                        
                    parts = parse_multipart(response.content, boundary)
                    if 'json' in parts and 'p2p' in parts:
                        meta = parts['json']['JpegPictureWithAppendData']
                        w, h = meta['jpegPicWidth'], meta['jpegPicHeight']
                        raw_data = parts['p2p']
                        p2p_len = meta['p2pDataLen']
                        
                        # Trích xuất ma trận nhiệt độ
                        matrix = np.frombuffer(raw_data[:p2p_len], dtype=np.float32).reshape(h, w)
                        
                        points = []
                        for pt in custom_points:
                            label = pt.get("label") or pt.get("id")
                            x = float(pt.get("x", 0.0))
                            y = float(pt.get("y", 0.0))
                            
                            # Quy đổi tọa độ tỷ lệ (0.0-1.0) sang index ma trận
                            mx = int(x * w)
                            my = int(y * h)
                            mx = max(0, min(w - 1, mx))
                            my = max(0, min(h - 1, my))
                            
                            temp = float(matrix[my, mx])
                            points.append({
                                "id": label,  # Sử dụng label làm định danh gửi về Dashboard
                                "temperature": round(temp, 1)
                            })
                        return points
                else:
                    print(f"[SENDER WARNING] HTTP {response.status_code} khi truy vấn P2P matrix. Chuyển sang dùng camera rules.", flush=True)
        except Exception as e:
            print(f"[SENDER ERROR] Lỗi khi xử lý ma trận nhiệt P2P: {e}. Thử fallback về camera rules.", flush=True)

    # 2. CHẾ ĐỘ MẶC ĐỊNH: TRUY VẤN CAMERA RULES (ID 1-6)
    try:
        response = requests.get(CAMERA_URL, auth=auth, timeout=5)
        if response.status_code == 200:
            data = response.json()
            rules_info = data.get("ThermometryRulesTemperatureInfoList", {}).get("ThermometryRulesTemperatureInfo", [])
            points = []
            for rule in rules_info:
                rule_id = rule.get("id")
                if rule_id in [1, 2, 3, 4, 5, 6]:
                    temp = rule.get("maxTemperature")
                    points.append({
                        "id": f"ID_{rule_id}",
                        "temperature": temp
                    })
            points.sort(key=lambda x: x["id"])
            return points
        else:
            print(f"[CAMERA ERROR] HTTP Status {response.status_code}", flush=True)
            return None
    except Exception as e:
        print(f"[CAMERA ERROR] Could not connect or parse: {e}", flush=True)
        return None

def send_to_server(points):
    """POST the thermal data to both local receiver and central server."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "timestamp": timestamp,
        "points": points
    }
    
    # 1. Send to local receiver to save history & generate predictions
    try:
        print(f"[SENDER] Sending data to local receiver: {payload}", flush=True)
        local_resp = requests.post(LOCAL_RECEIVER_URL, json=payload, timeout=10)
        if local_resp.status_code == 200:
            print(f"[SENDER SUCCESS] Local receiver processed successfully: {local_resp.text}", flush=True)
        else:
            print(f"[SENDER WARNING] Local receiver returned status {local_resp.status_code}: {local_resp.text}", flush=True)
    except Exception as e:
        print(f"[SENDER ERROR] Failed to send to local receiver: {e}", flush=True)

    # 2. Send to central server
    try:
        print(f"[SENDER] Sending data to central server: {payload}", flush=True)
        response = requests.post(SERVER_URL, json=payload, timeout=10)
        if response.status_code == 200:
            print(f"[SENDER SUCCESS] Central server response: {response.text}", flush=True)
            return True
        else:
            print(f"[SENDER ERROR] Central server returned status {response.status_code}: {response.text}", flush=True)
            return False
    except Exception as e:
        print(f"[SENDER ERROR] Failed to send to central server: {e}", flush=True)
        return False

def sync_targets_with_server():
    """Tự động đồng bộ danh sách targets từ Central Server về local receiver."""
    try:
        # 1. Gọi sang PC/Server để lấy danh sách điểm đang cấu hình ở file config.json
        base_url = "/".join(SERVER_URL.split("/")[:3]) # Lấy http://192.168.10.22:8100
        config_url = f"{base_url}/api/config"
        
        resp = requests.get(config_url, timeout=3.0)
        if resp.status_code != 200:
            return
            
        server_config = resp.json()
        server_targets = server_config.get("targets", [])
        if not server_targets:
            return
            
        # 2. Đọc config.json cục bộ
        config_json_path = os.path.join("timeseries_service", "model", "config.json")
        local_targets = []
        if os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                local_targets = cfg.get("targets", [])
                
        # 3. So sánh, nếu khác nhau thì cập nhật cục bộ qua API (skip forward)
        if sorted(local_targets) != sorted(server_targets):
            print(f"[SENDER] Phát hiện danh sách targets thay đổi trên Server: {server_targets}. Đang đồng bộ về local...", flush=True)
            local_update_url = "http://127.0.0.1:8080/api/config/update?forward=false"
            update_resp = requests.post(local_update_url, json={"targets": server_targets}, timeout=5)
            if update_resp.status_code == 200:
                print(f"[SENDER SUCCESS] Đã đồng bộ targets thành công về local receiver.", flush=True)
            else:
                print(f"[SENDER ERROR] Lỗi khi đồng bộ targets về local: {update_resp.text}", flush=True)
    except Exception as e:
        print(f"[SENDER WARNING] Không thể tự động đồng bộ cấu hình từ Server: {e}", flush=True)

def main():
    print(f"==================================================")
    print(f"   STARTING THERMAL DATA SENDER (EDGE SERVICE)    ")
    print(f"   Camera IP: {CAMERA_IP}                         ")
    print(f"   Server URL: {SERVER_URL}                       ")
    print(f"   Interval: {POLL_INTERVAL}s                     ")
    print(f"==================================================")

    # Đợi 3 giây để local receiver (uvicorn) kịp khởi động
    time.sleep(3)

    # Tự động đồng bộ targets từ Central Server lúc khởi chạy
    sync_targets_with_server()

    # Initial immediate check
    points = get_camera_temperatures()
    if points:
        print(f"[SENDER] Successfully fetched {len(points)} points from camera.", flush=True)
        send_to_server(points)
    else:
        print("[SENDER WARNING] Initial fetch failed or returned no data.", flush=True)

    # Main loop
    while True:
        try:
            time.sleep(POLL_INTERVAL)
            # Tự động đồng bộ targets từ Central Server trước khi quét
            sync_targets_with_server()
            
            points = get_camera_temperatures()
            if points:
                send_to_server(points)
            else:
                print("[SENDER WARNING] Skipping this cycle due to camera fetch error.", flush=True)
        except KeyboardInterrupt:
            print("\n[SENDER] Stopping service...", flush=True)
            break
        except Exception as e:
            print(f"[SENDER EXCEPTION] {e}", flush=True)
            time.sleep(10)

if __name__ == "__main__":
    main()
