import sys
import os
import subprocess
import time
import signal
import threading

# Đường dẫn tuyệt đối đến thư mục gốc của project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_SCRIPT = os.path.join(BASE_DIR, "yolo_detect.py")
THERMAL_SCRIPT = os.path.join(BASE_DIR, "timeseries_service", "thermal_receiver.py")
SENDER_SCRIPT = os.path.join(BASE_DIR, "thermal_sender.py")

processes = []

def forward_output(pipe, prefix):
    """Đọc liên tục từ pipe của tiến trình con để tránh tràn bộ đệm gây lag/treo."""
    try:
        for line in iter(pipe.readline, ''):
            if line:
                print(f"{prefix} {line.strip()}", flush=True)
    except Exception:
        pass

def signal_handler(sig, frame):
    print("\n[SYSTEM] Nhận tín hiệu tắt từ hệ thống. Đang dừng các tiến trình con...")
    for p in processes:
        try:
            if p.poll() is None:
                print(f"[SYSTEM] Đang dừng PID: {p.pid}...")
                p.terminate()
                p.wait(timeout=3)
        except Exception as e:
            print(f"[SYSTEM] Lỗi khi dừng tiến trình PID {p.pid}: {e}")
    print("[SYSTEM] Đã tắt tất cả các tiến trình con thành công.")
    sys.exit(0)

# Đăng ký signal để ngắt Ctrl+C hoặc khi hệ thống tắt dịch vụ
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def cleanup_old_processes():
    print("[INIT] Đang dọn dẹp các tiến trình cũ chạy ngầm...")
    # 1. Kill bất kỳ tiến trình nào đang chiếm dụng cổng 8080
    try:
        subprocess.run(["fuser", "-k", "8080/tcp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    
    # 2. Tìm và tắt các file python chạy ngầm cũ của hệ thống
    try:
        my_pid = os.getpid()
        # Tìm tất cả pid của python
        pids = subprocess.check_output(["pgrep", "-f", "python"]).decode().split()
        for pid_str in pids:
            try:
                pid = int(pid_str)
                if pid == my_pid:
                    continue
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace('\x00', ' ')
                if any(script in cmdline for script in ["yolo_detect.py", "thermal_sender.py", "thermal_receiver.py"]):
                    print(f"[INIT] Dừng tiến trình cũ PID {pid}: {cmdline}")
                    os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    except Exception:
        pass
    time.sleep(1.5)  # Chờ port được giải phóng hoàn toàn

def main():
    # Kiểm tra tham số dòng lệnh
    thermal_only = "--thermal-only" in sys.argv or "-t" in sys.argv

    print("=" * 70)
    if thermal_only:
        print("   KHỞI ĐỘNG HỆ THỐNG: CHỈ CHẠY DỊCH VỤ NHIỆT ĐỘ (THERMAL ONLY)      ")
    else:
        print("   KHỞI ĐỘNG HỆ THỐNG DUAL DAEOMON: YOLO DETECT + THERMAL RECEIVER   ")
    print("=" * 70)

    # Dọn dẹp tiến trình cũ trước khi chạy
    cleanup_old_processes()

    # 1. Khởi động Thermal Receiver API
    print("[INIT] Khởi động Thermal Receiver API...")
    thermal_cwd = os.path.join(BASE_DIR, "timeseries_service")
    p_thermal = subprocess.Popen(
        [sys.executable, "thermal_receiver.py"],
        cwd=thermal_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    processes.append(p_thermal)

    # 2. Khởi động YOLOv8 Headless Person Detection
    p_yolo = None
    if not thermal_only:
        print("[INIT] Khởi động YOLOv8 Headless Person Detection...")
        p_yolo = subprocess.Popen(
            [sys.executable, "-u", YOLO_SCRIPT],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        processes.append(p_yolo)
    else:
        print("[INIT] Đã tắt dịch vụ YOLO Person Detection (Thermal Only).")

    # 3. Khởi động Thermal Sender (Edge Service) (ĐÃ COMMENT - VÌ MÁY CHỦ SẼ GỬI DỮ LIỆU NHIỆT XUỐNG)
    p_sender = None
    # print("[INIT] Khởi động Thermal Sender...")
    # p_sender = subprocess.Popen(
    #     [sys.executable, "-u", SENDER_SCRIPT],
    #     cwd=BASE_DIR,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     text=True,
    #     bufsize=1
    # )
    # processes.append(p_sender)
    print("[INIT] Đã comment (tạm dừng) dịch vụ Thermal Sender (quét camera).")

    # 4. Tạo các luồng đọc log song song, không chặn (Non-blocking log forwarding)
    t_thermal = threading.Thread(target=forward_output, args=(p_thermal.stdout, "[THERMAL]"), daemon=True)
    t_thermal.start()
    
    if p_yolo:
        t_yolo = threading.Thread(target=forward_output, args=(p_yolo.stdout, "[YOLO]"), daemon=True)
        t_yolo.start()
        
    # if p_sender:
    #     t_sender = threading.Thread(target=forward_output, args=(p_sender.stdout, "[SENDER]"), daemon=True)
    #     t_sender.start()

    time.sleep(3)

    # Kiểm tra trạng thái khởi động ban đầu
    thermal_status = "ĐANG CHẠY" if p_thermal.poll() is None else "LỖI KHỞI ĐỘNG"
    yolo_status = "ĐANG CHẠY" if (p_yolo and p_yolo.poll() is None) else "TẮT"
    sender_status = "TẮT (ĐÃ COMMENT)"

    print("\n" + "-" * 70)
    print(f"[*] Thermal Receiver (Port 8080)   : {thermal_status}")
    print(f"[*] YOLOv8 Person Detect (Headless): {yolo_status}")
    print(f"[*] Thermal Sender (Edge Service)  : {sender_status}")
    print("-" * 70)
    print("Hệ thống đang chạy ngầm song song. Bấm Ctrl+C để dừng.\n")

    # Giám sát liên tục trạng thái sống của các tiến trình con
    try:
        while True:
            if p_thermal.poll() is not None:
                print(f"[CẢNH BÁO] Thermal Receiver đã dừng đột ngột với exit code {p_thermal.returncode}!")
                break

            if p_yolo and p_yolo.poll() is not None:
                print(f"[CẢNH BÁO] YOLOv8 Person Detect đã dừng đột ngột với exit code {p_yolo.returncode}!")
                break

            if p_sender and p_sender.poll() is not None:
                print(f"[CẢNH BÁO] Thermal Sender đã dừng đột ngột với exit code {p_sender.returncode}!")
                break

            time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
