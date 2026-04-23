import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import requests
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

ENGINE_PATH = "models/best.engine"
ONNX_PATH = "models/best.onnx"
PT_PATH = "models/best.pt"
ENGINE_META_PATH = f"{ENGINE_PATH}.meta.json"
AUTO_REBUILD_ENGINE = True
SOURCE = "rtsp://admin:Demo%402024@192.168.10.152:554/Streaming/Channels/102"

INPUT_W = 640
INPUT_H = 640

CONF_THRES = 0.55
PERSON_CLASS_ID = 0

MIN_BOX_W = 40
MIN_BOX_H = 80
MIN_BOX_AREA = 4000
MIN_ASPECT_RATIO = 0.20
MAX_ASPECT_RATIO = 1.20

DISPLAY_W = 960
DISPLAY_H = 540

# =========================
# API CONFIG
# =========================
API_URL = "http://127.0.0.1:8000/api/person-detection"
CAMERA_IP = "192.168.10.152"
MIN_API_INTERVAL = 5
API_TIMEOUT = 5
API_MAX_WORKERS = 1

# Ảnh gửi lên server sẽ nén jpeg với chất lượng này
JPEG_QUALITY = 85

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|max_delay;500000|fflags;nobuffer"
)

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)


def get_device_signature():
    device = cuda.Device(0)
    cc_major, cc_minor = device.compute_capability()
    return {
        "gpu_name": device.name(),
        "compute_capability": f"{cc_major}.{cc_minor}",
        "tensorrt_version": trt.__version__,
        "input_size": [INPUT_W, INPUT_H],
    }


def load_engine_metadata():
    if not os.path.exists(ENGINE_META_PATH):
        return None

    try:
        with open(ENGINE_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_engine_metadata(metadata):
    os.makedirs(os.path.dirname(ENGINE_META_PATH), exist_ok=True)
    with open(ENGINE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def engine_matches_current_device(metadata, current_device):
    if not metadata:
        return False

    keys = ("gpu_name", "compute_capability", "tensorrt_version", "input_size")
    return all(metadata.get(key) == current_device.get(key) for key in keys)


def rebuild_engine_from_onnx(current_device):
    if not os.path.exists(ONNX_PATH):
        return False

    trtexec_path = shutil.which("trtexec")
    if not trtexec_path:
        print("[TRT] Khong tim thay trtexec, bo qua rebuild engine tu ONNX.")
        return False

    base_command = [
        trtexec_path,
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={ENGINE_PATH}",
    ]

    print(f"[TRT] Dang rebuild engine FP16 cho GPU hien tai: {current_device['gpu_name']}")
    result = subprocess.run(base_command + ["--fp16"], check=False)
    if result.returncode != 0:
        print("[TRT] Build FP16 that bai, thu lai FP32.")
        result = subprocess.run(base_command, check=False)
    if result.returncode != 0:
        raise RuntimeError("Rebuild TensorRT engine tu ONNX that bai.")

    save_engine_metadata(current_device)
    print(f"[TRT] Da rebuild engine: {ENGINE_PATH}")
    return True


def rebuild_engine_from_pt(current_device):
    if not os.path.exists(PT_PATH):
        return False

    yolo_path = shutil.which("yolo")
    if not yolo_path:
        print("[TRT] Khong tim thay yolo CLI, bo qua export engine tu PT.")
        return False

    base_command = [
        yolo_path,
        "export",
        f"model={PT_PATH}",
        "format=engine",
        f"imgsz={INPUT_W}",
        "device=0",
    ]

    print(f"[TRT] Dang export TensorRT engine FP16 tu PT cho GPU hien tai: {current_device['gpu_name']}")
    result = subprocess.run(base_command + ["half=True"], check=False)
    if result.returncode != 0:
        print("[TRT] Export FP16 that bai, thu lai FP32.")
        result = subprocess.run(base_command + ["half=False"], check=False)
    if result.returncode != 0:
        raise RuntimeError("Export TensorRT engine tu best.pt that bai.")

    if not os.path.exists(ENGINE_PATH):
        raise RuntimeError(f"Ultralytics export xong nhung khong thay file {ENGINE_PATH}.")

    save_engine_metadata(current_device)
    print(f"[TRT] Da export engine: {ENGINE_PATH}")
    return True


def rebuild_engine(current_device):
    return rebuild_engine_from_onnx(current_device) or rebuild_engine_from_pt(current_device)


def prepare_engine():
    current_device = get_device_signature()
    metadata = load_engine_metadata()

    if not os.path.exists(ENGINE_PATH):
        if AUTO_REBUILD_ENGINE and rebuild_engine(current_device):
            return
        raise FileNotFoundError(
            f"Khong tim thay {ENGINE_PATH}. Hay dat file engine vao models/ "
            f"hoac dat {ONNX_PATH}/{PT_PATH} de script tu build lai."
        )

    if engine_matches_current_device(metadata, current_device):
        return

    if metadata:
        print(
            "[TRT] Engine hien tai duoc tao cho "
            f"{metadata.get('gpu_name')} / CC {metadata.get('compute_capability')} / "
            f"TensorRT {metadata.get('tensorrt_version')}, "
            "khac voi may dang chay."
        )
        if AUTO_REBUILD_ENGINE and rebuild_engine(current_device):
            return
    else:
        print(
            "[TRT] Chua co metadata cho engine, se thu load engine hien co. "
            "Neu load loi thi can rebuild engine tren dung GPU."
        )


prepare_engine()

with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    raise RuntimeError(
        "Khong load duoc TensorRT engine. Hay rebuild lai engine tren dung GPU hien tai."
    )

context = engine.create_execution_context()

input_name, output_name = None, None
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        input_name = name
    elif mode == trt.TensorIOMode.OUTPUT:
        output_name = name

input_shape = tuple(context.get_tensor_shape(input_name))
if -1 in input_shape:
    input_shape = (1, 3, INPUT_H, INPUT_W)
    context.set_input_shape(input_name, input_shape)

output_shape = tuple(context.get_tensor_shape(output_name))

d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(np.float32).itemsize)
output = np.empty(output_shape, dtype=np.float32)
stream = cuda.Stream()

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))


def preprocess(frame):
    h0, w0 = frame.shape[:2]
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img, h0, w0


def is_valid_person_box(x1, y1, x2, y2):
    bw = x2 - x1
    bh = y2 - y1
    area = bw * bh

    if bw < MIN_BOX_W or bh < MIN_BOX_H:
        return False

    if area < MIN_BOX_AREA:
        return False

    aspect_ratio = bw / (bh + 1e-6)
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False

    return True


def postprocess(output, frame, h0, w0):
    img_draw = frame.copy()
    preds = output[0]

    scale_x = w0 / INPUT_W
    scale_y = h0 / INPUT_H

    count = 0
    boxes = []

    for det in preds:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)

        if cls_id != PERSON_CLASS_ID:
            continue

        if score < CONF_THRES:
            continue

        x1 = int(max(0, min(w0 - 1, x1 * scale_x)))
        y1 = int(max(0, min(h0 - 1, y1 * scale_y)))
        x2 = int(max(0, min(w0 - 1, x2 * scale_x)))
        y2 = int(max(0, min(h0 - 1, y2 * scale_y)))

        if x2 <= x1 or y2 <= y1:
            continue

        if not is_valid_person_box(x1, y1, x2, y2):
            continue

        count += 1
        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": round(float(score), 4)
        })

        label = f"person: {score:.2f}"
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_draw,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return img_draw, count, boxes


def send_detection_api(person_count, boxes, frame_bgr):
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "timestamp": timestamp_str,
        "camera_ip": CAMERA_IP,
        "person_count": person_count,
        "alert_type": "person_detected",
        "boxes": boxes
    }

    # Encode ảnh thành jpeg trong RAM, không cần lưu file tạm
    success, encoded_img = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    if not success:
        print("[API] Loi encode anh")
        return

    files = {
        "image": (
            f"person_{int(time.time())}.jpg",
            encoded_img.tobytes(),
            "image/jpeg"
        )
    }

    data = {
        "metadata": json.dumps(payload, ensure_ascii=False)
    }

    try:
        response = requests.post(
            API_URL,
            data=data,
            files=files,
            timeout=API_TIMEOUT
        )
        print(f"[API] status={response.status_code} | response={response.text}")
    except Exception as e:
        print(f"[API] Loi gui API: {e}")


cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Không mở được RTSP: {SOURCE}")

cv2.namedWindow("TensorRT RTSP Test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("TensorRT RTSP Test", DISPLAY_W, DISPLAY_H)

prev_time = time.time()
last_api_sent_time = 0
last_has_person = False
api_executor = ThreadPoolExecutor(max_workers=API_MAX_WORKERS)
api_future = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ camera")
            break

        img, h0, w0 = preprocess(frame)

        cuda.memcpy_htod_async(d_input, img, stream)
        context.execute_async_v3(stream_handle=stream.handle)

        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        img_draw, count, boxes = postprocess(output, frame, h0, w0)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        has_person = count > 0
        should_send = False

        if has_person and not last_has_person:
            should_send = True
        elif has_person and (curr_time - last_api_sent_time >= MIN_API_INTERVAL):
            should_send = True

        api_busy = api_future is not None and not api_future.done()
        if should_send and not api_busy:
            api_future = api_executor.submit(send_detection_api, count, boxes, img_draw.copy())
            last_api_sent_time = curr_time

        last_has_person = has_person

        cv2.putText(
            img_draw,
            f"FPS: {fps:.2f} | Persons: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        display_frame = cv2.resize(img_draw, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("TensorRT RTSP Test", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite("result_camera.jpg", img_draw)
            print("Da luu: result_camera.jpg")

finally:
    api_executor.shutdown(wait=False, cancel_futures=True)
    cap.release()
    cv2.destroyAllWindows()
