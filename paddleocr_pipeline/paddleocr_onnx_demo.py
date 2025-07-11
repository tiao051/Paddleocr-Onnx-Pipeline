import cv2
import numpy as np
import onnxruntime as ort
import os
from db_postprocess import DBPostProcess

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from paddleocr_pipeline

# Load models
det_sess = ort.InferenceSession(os.path.join(project_root, "models", "det_model.onnx"))
rec_sess = ort.InferenceSession(os.path.join(project_root, "models", "rec_model.onnx"))

# Load image
img = cv2.imread(os.path.join(project_root, "test", "test.jpg"))
if img is None:
    # Create a test image if not found
    img = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello World", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "PaddleOCR Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "ONNX Model", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    os.makedirs(os.path.join(project_root, "test"), exist_ok=True)
    cv2.imwrite(os.path.join(project_root, "test", "test.jpg"), img)

ori_img = img.copy()
orig_h, orig_w = img.shape[:2]

### 1. Preprocess for detection
def resize_det(img, target_size=960):
    h, w, _ = img.shape
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    new_h = max(32, int(np.round(new_h / 32)) * 32)
    new_w = max(32, int(np.round(new_w / 32)) * 32)
    resized_img = cv2.resize(img, (new_w, new_h))
    norm_img = resized_img.astype(np.float32) / 255.
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    norm_img = (norm_img - mean) / std
    chw_img = norm_img.transpose(2, 0, 1)
    return chw_img[np.newaxis, ...].astype(np.float32), (new_h, new_w)

# 1. Resize input
input_tensor, (input_h, input_w) = resize_det(img)

### 2. Run detection model
det_out = det_sess.run(None, {"x": input_tensor})[0]  # (1, 1, H, W)
outs_dict = {"maps": det_out}

# 3. Postprocess
ratio_h = input_h / float(orig_h)
ratio_w = input_w / float(orig_w)
shape_list = [[orig_h, orig_w, ratio_h, ratio_w]]
postprocess = DBPostProcess()
boxes_result = postprocess(outs_dict, shape_list)
filtered_boxes = boxes_result[0]['points']

### 4. Recognition preprocess (improved)
def preprocess_crop(img, box):
    # Get 4 corner points and apply perspective transform
    box = np.array(box).astype(np.float32)
    
    # Get bounding rectangle
    rect = cv2.boundingRect(box)
    x, y, w, h = rect
    
    # Simple crop for now (can be improved with perspective transform)
    crop = img[y:y+h, x:x+w]
    
    # Resize to standard recognition input size
    crop = cv2.resize(crop, (320, 48))  # Standard PaddleOCR rec input
    
    # Normalize
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Change to CHW format
    crop = crop.transpose(2, 0, 1)
    return crop[np.newaxis, ...].astype(np.float32)

### 5. Proper CTC decode for PaddleOCR
def decode_rec_output(preds):
    """
    Debug decode to understand what model is outputting
    """
    print(f"=== DEBUG RECOGNITION OUTPUT ===")
    print(f"Prediction shape: {preds.shape}")
    print(f"Min value: {np.min(preds):.4f}, Max value: {np.max(preds):.4f}")
    
    # Get predictions
    preds_idx = np.argmax(preds, axis=2)[0]  # (seq_len,)
    print(f"Predicted indices: {preds_idx}")
    print(f"Unique indices: {np.unique(preds_idx)}")
    print(f"Max index: {np.max(preds_idx)}")
    
    # Current charset
    charset = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    charset = "#" + charset  # # is blank token
    print(f"Charset length: {len(charset)}")
    
    # Show what each index maps to
    result = []
    prev_idx = -1
    
    for i, idx in enumerate(preds_idx):
        if idx < len(charset):
            char = charset[idx]
            print(f"Position {i}: index {idx} -> '{char}'")
        else:
            print(f"Position {i}: index {idx} -> OUT OF RANGE!")
            
        if idx != prev_idx and idx != 0:  # 0 is blank
            if idx < len(charset):
                result.append(charset[idx])
        prev_idx = idx
    
    decoded = ''.join(result)
    print(f"Final decoded: '{decoded}'")
    print("=== END DEBUG ===\n")
    
    return decoded

### 6. Run recognition per box
print(f"Found {len(filtered_boxes)} text boxes")

for i, box in enumerate(filtered_boxes):
    print(f"Processing box {i+1}/{len(filtered_boxes)}")
    
    try:
        crop_input = preprocess_crop(ori_img, box)
        print(f"Crop shape: {crop_input.shape}")
        
        rec_output = rec_sess.run(None, {"x": crop_input})[0]
        print(f"Recognition output shape: {rec_output.shape}")
        
        text = decode_rec_output(rec_output)
        print(f"Recognized text: '{text}'")
        
        # Draw box and text
        pts = np.array(box).astype(np.int32)
        cv2.polylines(ori_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Put text above the box
        x, y = pts[0]
        cv2.putText(ori_img, text, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    except Exception as e:
        print(f"Error processing box {i+1}: {e}")
        continue

### 7. Show result
cv2.imshow("Result", ori_img)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()