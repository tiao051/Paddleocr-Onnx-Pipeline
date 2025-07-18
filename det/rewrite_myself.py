import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import datetime
import os
from det.postprocessing_onnx import DBPostProcessONNX

def resize_and_normalize_ppocrv5(img, target_size=640):
    """
    PP-OCRv5 detection preprocessing - based on det_pp-ocrv5.yml config
    YAML config: d2s_train_image_shape: [3, 640, 640]
    """
    original_h, original_w = img.shape[:2]
    ratio = target_size / max(original_h, original_w)
    new_h = int(original_h * ratio)
    new_w = int(original_w * ratio)

    # Round to nearest multiple of 32 (required by model architecture)
    new_h = max(32, int(np.round(new_h / 32)) * 32)
    new_w = max(32, int(np.round(new_w / 32)) * 32)

    resized_img = cv2.resize(img, (new_w, new_h))

    # Normalize (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm_img = resized_img.astype(np.float32) / 255.0
    normalized = (norm_img - mean) / std

    # HWC -> CHW
    chw = normalized.transpose(2, 0, 1)
    input_tensor = chw[np.newaxis, ...].astype(np.float32)

    # scale ratios for postprocess (resize back)
    scale_h = new_h / original_h
    scale_w = new_w / original_w
    return input_tensor, (scale_h, scale_w), resized_img

def analyze_detection_output(prob_map):
    """
    Analyze detection output to understand model behavior
    """
    print(f"\n🔬 Analyzing detection output:")
    print(f"   Shape: {prob_map.shape}")
    print(f"   Min: {prob_map.min():.4f}")
    print(f"   Max: {prob_map.max():.4f}")
    print(f"   Mean: {prob_map.mean():.4f}")
    print(f"   Std: {prob_map.std():.4f}")
    
    # Percentile analysis
    percentiles = [5, 25, 50, 75, 95]
    values = [np.percentile(prob_map, p) for p in percentiles]
    print(f"   Percentiles: {dict(zip(percentiles, [f'{v:.4f}' for v in values]))}")
    
    # Histogram analysis
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        above_thresh = (prob_map > thresh).sum()
        percentage = (above_thresh / prob_map.size) * 100
        print(f"   > {thresh:.1f}: {above_thresh:6d} pixels ({percentage:5.1f}%)")
    
    return prob_map.min(), prob_map.max(), prob_map.mean()

def run_detection_onnx(input_tensor: np.ndarray, model_path: str) -> np.ndarray:
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if input_tensor.dtype != np.float32:
        input_tensor = input_tensor.astype(np.float32)
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, axis=0)

    outputs = session.run(None, {input_name: input_tensor})
    return outputs[0]  # shape: (1, 1, H, W)

def visualize_prob_map(prob_map: np.ndarray):
    plt.figure(figsize=(10, 6))
    plt.imshow(prob_map, cmap='gray')
    plt.title("Detection Probability Map")
    plt.colorbar()
    plt.axis("off")
    plt.show()

def visualize_detection_results(img: np.ndarray, boxes: list, scores: list, save_only=False):
    """
    Visualize detection results by drawing bounding boxes on image
    """
    result_img = img.copy()
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        # Convert box to numpy array if needed
        if isinstance(box, list):
            box = np.array(box)
        
        # Draw bounding box
        pts = box.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)
        
        # Add confidence score text
        cv2.putText(result_img, f"{score:.3f}", 
                   (int(box[0][0]), int(box[0][1]) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save result to output directory
    os.makedirs("output", exist_ok=True)
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
    filename = now.strftime("detection_result_%Y%m%d_%H%M%S.jpg")
    output_path = os.path.join("output", filename)
    cv2.imwrite(output_path, result_img)
    print(f"   Result saved to: {output_path}")
    
    # Display result only if not save_only
    if not save_only:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Text Detection Results - {len(boxes)} boxes detected")
        plt.axis("off")
        plt.show()
    
    return result_img

def main(test_image="eng_test1.jpg"):
    """
    Complete OCR detection pipeline:
    Image → Preprocessing → ONNX Detection → Postprocessing → Final Boxes
    """
    # Dynamic path based on images folder
    image_path = f"images/{test_image}"
    
    # Use generic model name
    model_path = "models/det_model.onnx"

    print(f"🚀 Starting complete OCR detection pipeline...")
    print(f"   📁 Model: {os.path.basename(model_path)}")
    print(f"   📸 Image: {test_image}")
    print("=" * 60)

    # Step 1: Load image
    print("📸 Step 1: Loading image...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh tại đường dẫn: {image_path}")
    
    original_h, original_w = img.shape[:2]
    print(f"   Original image size: {original_w}x{original_h}")

    # Step 2: Preprocess
    print("\n🔧 Step 2: Preprocessing image...")
    input_tensor, (scale_h, scale_w), resized_img = resize_and_normalize_ppocrv5(img)
    print(f"   Input tensor shape: {input_tensor.shape}")
    print(f"   Scale ratios: height={scale_h:.3f}, width={scale_w:.3f}")

    # Step 3: Run detection ONNX
    print("\n🧠 Step 3: Running ONNX detection model...")
    pred_map = run_detection_onnx(input_tensor, model_path)
    prob_map = pred_map[0, 0]  # (1, 1, H, W) -> (H, W)
    print(f"   Probability map shape: {prob_map.shape}")
    print(f"   Probability range: [{np.min(prob_map):.3f}, {np.max(prob_map):.3f}]")
    
    # Analyze detection output for debugging
    analyze_detection_output(prob_map)

    # Step 4: Initialize postprocessor
    print("\n⚙️  Step 4: Initializing DB postprocessor...")
    
    # Use standard thresholds
    thresh = 0.3
    box_thresh = 0.6
    print(f"   🔧 Using standard thresholds")
    
    postprocessor = DBPostProcessONNX(
        thresh=thresh,        # Standard threshold
        box_thresh=box_thresh, # Standard confidence threshold
        max_candidates=1000,  # Max contours (matches YAML)
        unclip_ratio=1.5,     # Box expansion (matches YAML: 1.5)
        score_mode="fast",    # Fast scoring method
        box_type='quad'       # Quadrilateral boxes
    )
    print(f"   Binary threshold: {postprocessor.thresh}")
    print(f"   Confidence threshold: {postprocessor.box_thresh}")
    print(f"   Unclip ratio: {postprocessor.unclip_ratio}")

    # Step 5: Run postprocessing
    print("\n🔍 Step 5: Running DB postprocessing...")
    shape_info = [scale_h, scale_w]  # Pass scale ratios for coordinate conversion
    boxes, scores = postprocessor(pred_map, shape_info)
    
    print(f"   Detected boxes: {len(boxes)}")
    if len(boxes) > 0:
        print(f"   Confidence scores: {[f'{s:.3f}' for s in scores]}")
        
        # Print detailed box information
        print(f"\n📦 Detected text boxes:")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if isinstance(box, np.ndarray):
                box_coords = box.flatten()
            else:
                box_coords = np.array(box).flatten()
            print(f"   Box {i+1}: [{', '.join([f'{x:.1f}' for x in box_coords])}]")
            print(f"           Score: {score:.4f}")
            
            # Calculate box size
            width = abs(box_coords[2] - box_coords[0])
            height = abs(box_coords[5] - box_coords[1])
            print(f"           Size: {width:.1f}x{height:.1f} pixels")
    else:
        print("   ⚠️  No text boxes detected")

    # Step 6: Visualize results
    print("\n📊 Step 6: Visualizing results...")
    
    # Show detection results on original image
    if len(boxes) > 0:
        print("   Drawing detection results and saving to file...")
        result_img = visualize_detection_results(img, boxes, scores, save_only=True)
    
    print("\n✅ Pipeline completed successfully!")
    print("=" * 60)
    
    return {
        'image': img,
        'probability_map': prob_map,
        'boxes': boxes,
        'scores': scores,
        'shape_info': shape_info
    }

def main_det_run(test_image="eng_test1.jpg"):
    """Main detection function called by main.py"""
    try:
        result = main(test_image)
        print(f"✅ Successfully processed image with {len(result['boxes'])} text regions detected!")
        return result['image'], result['boxes']  # ✅ Return image + boxes
    except Exception as e:
        print(f"⚠️  Error with real image: {e}")
        return None, None 

