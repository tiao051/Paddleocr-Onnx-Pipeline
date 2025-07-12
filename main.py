import det.rewrite_myself as det
import numpy as np
import os
import cv2
import ctypes
from datetime import datetime
import utils.crop as crop  # Import crop module dùng trong crop

def get_timestamped_filename(prefix="det_crop_", ext=".jpg"):
    """Tạo tên file theo timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}{ext}"

def run_det():
    print("[1/3] Running detection...")
    img, boxes = det.main_det_run()  # ✅ nhận cả ảnh và box từ model
    print(f"[1/3] Found {len(boxes)} text boxes.")
    return img, boxes

def run_crop(img, raw_boxes, save_debug=True):
    print("[2/3] Cropping regions...")

    print("🔍 Debug: Reordering boxes...")
    sorted_boxes = crop.sort_boxes_top_to_bottom_left_to_right(raw_boxes)

    ordered_boxes = []
    for idx, box in enumerate(sorted_boxes):
        raw = np.array(box).reshape(4, 2)
        ordered = crop.order_points_clockwise(raw.copy())
        ordered_boxes.append(ordered)

    # 🔍 Vẽ box lên ảnh gốc để lưu file debug
    img_with_boxes = img.copy()
    for idx, box in enumerate(ordered_boxes):
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(img_with_boxes, str(idx + 1), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if save_debug:
        debug_filename = get_timestamped_filename()
        cv2.imwrite(debug_filename, img_with_boxes)
        print(f"🖼️  Debug image saved to: {debug_filename}")

    # ✂️ Crop từng vùng
    cropped_imgs = []
    print("\n✂️ Cropping each region...")
    for idx, box in enumerate(ordered_boxes):
        try:
            crop_img = crop.get_rotate_crop_image(img, box)
            cropped_imgs.append(crop_img)
            print(f"  ✅ Cropped Box {idx + 1} - Size: {crop_img.shape[1]}x{crop_img.shape[0]}")
        except Exception as e:
            print(f"  ❌ Failed to crop Box {idx + 1}: {str(e)}")

    print(f"\n[2/3] Cropped {len(cropped_imgs)} text regions successfully.")
    return cropped_imgs

def save_crops(crops, output_dir="output_crops"):
    os.makedirs(output_dir, exist_ok=True)
    for idx, crop in enumerate(crops):
        cv2.imwrite(os.path.join(output_dir, f"crop_{idx}.jpg"), crop)
    print(f"[3/3] Crops saved to: {output_dir}")

def main():
    # Detection
    img, boxes = run_det()

    # Crop
    crops = run_crop(img, boxes)

    # Save crops
    save_crops(crops)

if __name__ == "__main__":
    main()
