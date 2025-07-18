import det.rewrite_myself as det
import numpy as np
import os
import cv2
from datetime import datetime
import utils.crop as crop
from rec.rec_inference_onnx import RecognitionONNX

# Global variables for model and image selection
SELECTED_MODEL_VERSION = "v4"
SELECTED_TEST_IMAGE = "test1.jpg"

def choose_model_version():
    """Interactive model version selection"""
    print("\n🔧 Choose Model Version:")
    print("1. PP-OCRv4 Models (det_model_v4.onnx, rec_model_v4.onnx)")
    print("2. PP-OCRv5 Models (det_model.onnx, rec_model.onnx)")
    
    while True:
        try:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice == "1":
                return "v4"
            elif choice == "2":
                return "v5"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n\n❌ Program cancelled by user.")
            exit(0)

def choose_test_image():
    """Interactive test image selection"""
    test_dir = "images"
    
    # Get available test images
    if not os.path.exists(test_dir):
        print(f"⚠️  Test directory '{test_dir}' not found!")
        return "test.jpg"  # fallback
    
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"⚠️  No image files found in '{test_dir}' directory!")
        return "test.jpg"  # fallback
    
    print(f"\n📸 Available Test Images in '{test_dir}':")
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_dir, img_file)
        if os.path.exists(img_path):
            # Get image size
            try:
                img = cv2.imread(img_path)
                h, w = img.shape[:2]
                file_size = os.path.getsize(img_path) / 1024  # KB
                print(f"{i}. {img_file} ({w}x{h}, {file_size:.1f}KB)")
            except:
                print(f"{i}. {img_file} (unable to read dimensions)")
        else:
            print(f"{i}. {img_file} (file not found)")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(image_files)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(image_files):
                return image_files[choice_idx]
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(image_files)}.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\n❌ Program cancelled by user.")
            exit(0)

def get_model_paths(version):
    """Get model paths based on version"""
    if version == "v4":
        return {
            "det_model": "models/det_model_v4.onnx",
            "rec_model": "models/rec_model_v4.onnx"
        }
    elif version == "v5":
        return {
            "det_model": "models/det_model.onnx",  
            "rec_model": "models/rec_model.onnx"  
        }
    else:
        raise ValueError(f"Unknown model version: {version}")

def setup_test_configuration():
    """Setup test configuration - choose model and image"""
    global SELECTED_MODEL_VERSION, SELECTED_TEST_IMAGE
    
    print("🚀 PP-OCR ONNX Pipeline Configuration")
    print("=" * 50)
    
    # Choose model version
    SELECTED_MODEL_VERSION = choose_model_version()
    print(f"✅ Selected model version: PP-OCR{SELECTED_MODEL_VERSION}")
    
    # Choose test image
    SELECTED_TEST_IMAGE = choose_test_image()
    print(f"✅ Selected test image: {SELECTED_TEST_IMAGE}")
    
    # Verify model files exist
    model_paths = get_model_paths(SELECTED_MODEL_VERSION)
    missing_models = []
    
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
        else:
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"   📁 {model_name}: {model_path} ({file_size:.1f}MB)")
    
    if missing_models:
        print("\n❌ Missing model files:")
        for missing in missing_models:
            print(f"   - {missing}")
        print("\n🔧 Please ensure all model files are present before running.")
        exit(1)
    
    # Verify test image exists
    test_image_path = os.path.join("images", SELECTED_TEST_IMAGE)
    if not os.path.exists(test_image_path):
        print(f"\n❌ Test image not found: {test_image_path}")
        exit(1)
    else:
        print(f"   📸 Test image: {test_image_path}")
    
    print(f"\n✅ Configuration complete!")
    print("=" * 50)

def get_timestamped_filename(prefix="det_crop_", ext=".jpg", output_dir="output"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{prefix}{timestamp}{ext}")

def run_det():
    print("[1/4] Running detection...")
    # Pass selected configuration to detection module
    img, boxes = det.main_det_run(SELECTED_TEST_IMAGE)
    print(f"[1/4] Found {len(boxes)} text boxes.")
    return img, boxes

def run_crop(img, raw_boxes, save_debug=True):
    print("[2/4] Cropping regions...")

    # Handle empty boxes
    if len(raw_boxes) == 0:
        print("   ⚠️  No boxes to crop!")
        return [], []

    print("🔍 Sorting and ordering boxes...")
    sorted_boxes = crop.sort_boxes_top_to_bottom_left_to_right(raw_boxes)

    ordered_boxes = []
    for box in sorted_boxes:
        raw = np.array(box).reshape(4, 2)
        ordered = crop.order_points_clockwise(raw.copy())
        ordered_boxes.append(ordered)

    # Vẽ box lên ảnh gốc và lưu vào output
    if save_debug:
        img_with_boxes = img.copy()
        for idx, box in enumerate(ordered_boxes):
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_with_boxes, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(img_with_boxes, str(idx + 1), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        debug_filename = get_timestamped_filename()
        cv2.imwrite(debug_filename, img_with_boxes)
        print(f"🖼️  Debug image saved to: {debug_filename}")

    # Crop ảnh (không lưu từng crop nữa)
    cropped_imgs = []
    for idx, box in enumerate(ordered_boxes):
        try:
            crop_img = crop.get_rotate_crop_image(img, box)
            cropped_imgs.append(crop_img)
            print(f"  ✅ Cropped region {idx+1}")

        except Exception as e:
            print(f"  ❌ Failed to crop: {str(e)}")

    print(f"[2/4] Cropped {len(cropped_imgs)} regions.")
    return ordered_boxes, cropped_imgs

def run_recognition(crops):
    print("[3/4] Running recognition...")
    
    # Handle empty crops
    if len(crops) == 0:
        print("   ⚠️  No cropped images to recognize!")
        return []
    
    # Get correct model path based on selected version
    model_paths = get_model_paths(SELECTED_MODEL_VERSION)
    model_path = model_paths["rec_model"]
    
    print(f"   🧠 Using {SELECTED_MODEL_VERSION} recognition model: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Recognition model not found: {model_path}")

    recognizer = RecognitionONNX(model_path)
    results = recognizer.recognize_batch(crops)

    print(f"[3/4] Recognized {len(results)} texts.")
    return results  # List of (text, score)

def format_output(boxes, texts):
    output = []
    for box, text_result in zip(boxes, texts):
        if isinstance(text_result, tuple):
            text, score = text_result
        else:
            text = text_result
            score = 1.0  # default confidence
        output.append([box.tolist(), text, round(float(score), 5)])
    return output

def draw_ocr_result_on_blank_canvas(final_output, canvas_size=None, output_path=None):
    """
    Vẽ kết quả OCR từ final_output lên ảnh trắng giống PaddleOCR
    """
    # Handle empty output
    if len(final_output) == 0:
        print("   ⚠️  No OCR results to draw!")
        return None
        
    # Tự động tính canvas size dựa trên tọa độ boxes
    if canvas_size is None:
        all_points = []
        for box, _, _ in final_output:
            all_points.extend(box)

        max_x = max(point[0] for point in all_points) + 50  # Thêm padding
        max_y = max(point[1] for point in all_points) + 50  # Thêm padding
        canvas_size = (int(max_y), int(max_x))  # (height, width)

    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255

    print("🔍 Drawing OCR results...")
    print(f"  Canvas size: {canvas_size} (H×W)")

    for i, (box, text, score) in enumerate(final_output):
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        color_box = (0, 200, 255)        # Light blue
        color_text = (0, 0, 0)           # Black
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Tính vị trí text bên trong box
        x_min = min(point[0] for point in box)
        y_min = min(point[1] for point in box)
        y_max = max(point[1] for point in box)

        # Vẽ text bên trong box, căn giữa theo chiều cao
        text_x = int(x_min + 5)  # Thêm padding từ cạnh trái
        text_y = int(y_min + (y_max - y_min) / 2 + 5)  # Căn giữa theo chiều cao

        # Draw polygon box
        cv2.polylines(canvas, [pts], isClosed=True, color=color_box, thickness=2)

        # Draw text bên trong box
        font_scale = 0.5
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, color_text, 1, cv2.LINE_AA)

    if output_path:
        cv2.imwrite(output_path, canvas)
        print(f"🖼️  OCR visualization saved to: {output_path}")

    return canvas

def main():
    # Setup configuration first
    setup_test_configuration()
    
    # Run OCR pipeline
    img, boxes = run_det()
    ordered_boxes, crops = run_crop(img, boxes)
    texts = run_recognition(crops)
    final_output = format_output(ordered_boxes, texts)

    print("\n[4/4] 🔍 Final OCR Result (PaddleOCR format):")
    for line in final_output:
        print(line)

    # 🖼️ Vẽ lại kết quả OCR lên ảnh trắng và lưu vào output
    output_img_path = get_timestamped_filename(prefix="ocr_result_", ext=".jpg")
    draw_ocr_result_on_blank_canvas(final_output, output_path=output_img_path)

if __name__ == "__main__":
    main()