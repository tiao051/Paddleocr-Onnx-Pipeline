import cv2
import numpy as np


import cv2
import numpy as np

def resize_and_normalize_ppocrv5(img, target_size=960):
    """
    Preprocessing cho detection model của PP-OCRv5 (dùng DB).
    Debug theo input: img.shape = (720, 1280, 3)
    Note: model sẽ nhận một input gốc từ người dùng (img.shape) và khi đi vào hàm resize
    mục đích để cho ra 2 biến: 1. là input_tensor (đầu vào của model), 2. là shape_info (để làm tham số cho phần
    scale ngược lại từ image đã qua xử lý, để có thể detect được vị trí đó trong img đã xử lý cụ thể là ở đâu trong input gốc))
    """

    # Bước 1: Lấy kích thước gốc
    original_h, original_w = img.shape[:2]
    # original_h = 720, original_w = 1280

    # Bước 2: Tính scale sao cho max(h, w) = 960
    scale = float(target_size) / max(original_h, original_w)
    # scale = 960 / 1280 = 0.75

    new_h = int(original_h * scale)
    new_w = int(original_w * scale)
    # new_h = int(720 * 0.75) = 540
    # new_w = int(1280 * 0.75) = 960

    # Bước 3: Làm tròn về bội số của 32
    new_h = max(32, int(np.round(new_h / 32)) * 32)
    new_w = max(32, int(np.round(new_w / 32)) * 32)
    # new_h = round(540 / 32) * 32 = 17 * 32 = 544
    # new_w = round(960 / 32) * 32 = 30 * 32 = 960

    # Kết quả resize: (544, 960)
    resized_img = cv2.resize(img, (new_w, new_h))
    # resized_img.shape = (544, 960, 3)

    # Bước 4: Normalize theo ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_float = resized_img.astype(np.float32) / 255.0
    # Giả sử pixel tại (0,0): [120, 135, 150] → [0.4706, 0.5294, 0.5882]

    # Normalize pixel (ví dụ 1 pixel):
    #   R: (0.4706 - 0.485) / 0.229 ≈ -0.0627
    #   G: (0.5294 - 0.456) / 0.224 ≈  0.3276
    #   B: (0.5882 - 0.406) / 0.225 ≈  0.809

    normalized = (img_float - mean) / std
    # normalized.shape = (544, 960, 3)

    # Bước 5: HWC → CHW
    chw = normalized.transpose(2, 0, 1)
    # chw.shape = (3, 544, 960)

    # Bước 6: Add batch dimension → NCHW
    input_tensor = chw[np.newaxis, ...].astype(np.float32)
    # input_tensor.shape = (1, 3, 544, 960)

    # Bước 7: Tính shape_info để postprocess
    ratio_h = new_h / original_h  # 544 / 720 ≈ 0.7556
    ratio_w = new_w / original_w  # 960 / 1280 = 0.75
    shape_info = [ratio_h, ratio_w]

    return input_tensor, shape_info

def preprocess_crop_for_rec(img, box, target_size=(320, 48)):
    """
    Preprocessing cho recognition model (cắt + normalize box).
    """
    box = np.array(box).astype(np.float32)
    x, y, w, h = cv2.boundingRect(box)
    cropped = img[y:y + h, x:x + w]

    resized = cv2.resize(cropped, target_size)
    img_float = resized.astype(np.float32) / 255.0
    normalized = (img_float - 0.5) / 0.5

    chw = normalized.transpose(2, 0, 1)
    return chw[np.newaxis, ...].astype(np.float32)


def demo_preprocess():
    """
    Demo preprocess cho PP-OCRv5 detection + recognition
    """
    print("PP-OCRv5 Preprocessing Demo")

    # Tạo ảnh test
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "PADDLE OCR", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Detection
    det_input, shape_info = resize_and_normalize_ppocrv5(img, target_size=960)
    print(f"[Detection] Input shape: {det_input.shape}")
    print(f"[Detection] Resize ratios: {shape_info}")

    # Recognition (fake box)
    box = [[100, 100], [300, 100], [300, 200], [100, 200]]
    rec_input = preprocess_crop_for_rec(img, box)
    print(f"[Recognition] Input shape: {rec_input.shape}")


if __name__ == "__main__":
    demo_preprocess()
