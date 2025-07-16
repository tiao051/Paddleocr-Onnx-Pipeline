"""
PP-OCR ONNX vs PaddleOCR Official Comparison Script
So sánh kết quả giữa ONNX pipeline tự viết và PaddleOCR chính thức
"""

import os
import cv2
import numpy as np
import time
from datetime import datetime
import json

# Import custom ONNX pipeline
import det.rewrite_myself as det_onnx
from rec.rec_inference_onnx import RecognitionONNX
import utils.crop as crop

def install_paddleocr_if_needed():
    """Install PaddleOCR if not available"""
    try:
        import paddleocr
        return True
    except ImportError:
        print("📦 PaddleOCR not found. Installing...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "paddleocr"])
            import paddleocr
            return True
        except Exception as e:
            print(f"❌ Failed to install PaddleOCR: {e}")
            return False

def run_onnx_pipeline(image_path, model_version="v4"):
    """Run custom ONNX pipeline"""
    print(f"\n🔧 Running ONNX Pipeline (v{model_version})...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Get model paths
    if model_version == "v4":
        det_model = "models/det_model_v4.onnx"
        rec_model = "models/rec_model_v4.onnx"
    elif model_version == "v5":
        det_model = "models/det_model_v5.onnx"
        rec_model = "models/rec_model_v5.onnx"
    else:
        raise ValueError(f"Unsupported model version: {model_version}")
    
    # Check if models exist
    if not os.path.exists(det_model):
        print(f"❌ Detection model not found: {det_model}")
        return None
    if not os.path.exists(rec_model):
        print(f"❌ Recognition model not found: {rec_model}")
        return None
    
    try:
        # Step 1: Detection using det.rewrite_myself.main_det_run
        print("🔍 [1/3] Running detection...")
        img, boxes = det_onnx.main_det_run(model_version, os.path.basename(image_path))
        det_time = time.time() - start_time
        print(f"   ✅ Found {len(boxes)} text boxes in {det_time:.3f}s")
        
        if len(boxes) == 0:
            print("   ⚠️  No text detected!")
            return {
                'boxes': [],
                'texts': [],
                'scores': [],
                'full_results': [],
                'detection_time': det_time,
                'recognition_time': 0,
                'total_time': det_time
            }
        
        # Step 2: Crop regions
        print("🖼️  [2/3] Cropping text regions...")
        crop_start = time.time()
        
        # Read original image if not already loaded
        if img is None:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
        
        # Crop all detected regions
        cropped_images = []
        for i, box in enumerate(boxes):
            try:
                # Convert box to proper format for cropping
                if len(box) == 8:  # [x1,y1,x2,y2,x3,y3,x4,y4]
                    box_points = np.array(box, dtype=np.float32).reshape(4, 2)
                else:
                    box_points = np.array(box, dtype=np.float32)
                
                # Ensure points are valid 4x2 array
                if box_points.shape != (4, 2):
                    print(f"   ❌ Invalid box shape for region {i+1}: {box_points.shape}")
                    cropped_images.append(None)
                    continue
                
                cropped = crop.get_rotate_crop_image(img, box_points)
                if cropped is not None and cropped.size > 0:
                    cropped_images.append(cropped)
                    print(f"   ✅ Cropped region {i+1}: {cropped.shape}")
                else:
                    print(f"   ⚠️  Failed to crop region {i+1}")
                    cropped_images.append(None)
            except Exception as e:
                print(f"   ❌ Error cropping region {i+1}: {e}")
                cropped_images.append(None)
        
        crop_time = time.time() - crop_start
        
        # Step 3: Recognition
        print("🧠 [3/3] Running recognition...")
        rec_start = time.time()
        
        # Initialize recognition model
        recognizer = RecognitionONNX(rec_model)
        
        texts = []
        scores = []
        
        for i, cropped_img in enumerate(cropped_images):
            if cropped_img is not None:
                try:
                    text = recognizer.recognize(cropped_img)
                    score = 1.0  # ONNX recognition doesn't return confidence score
                    texts.append(text)
                    scores.append(score)
                    print(f"   ✅ Region {i+1}: '{text}' (confidence: {score:.3f})")
                except Exception as e:
                    print(f"   ❌ Recognition failed for region {i+1}: {e}")
                    texts.append("")
                    scores.append(0.0)
            else:
                texts.append("")
                scores.append(0.0)
        
        rec_time = time.time() - rec_start
        total_time = time.time() - start_time
        
        # Format results like PaddleOCR
        full_results = []
        for box, text, score in zip(boxes, texts, scores):
            if len(box) == 8:  # [x1,y1,x2,y2,x3,y3,x4,y4]
                box_formatted = [[box[0], box[1]], [box[2], box[3]], 
                               [box[4], box[5]], [box[6], box[7]]]
            else:
                box_formatted = box.tolist() if hasattr(box, 'tolist') else box
            
            full_results.append([box_formatted, text, score])
        
        print(f"✅ ONNX Pipeline completed in {total_time:.3f}s")
        print(f"   Detection: {det_time:.3f}s, Recognition: {rec_time:.3f}s")
        
        return {
            'boxes': boxes,
            'texts': texts, 
            'scores': scores,
            'full_results': full_results,
            'detection_time': det_time,
            'recognition_time': rec_time,
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"❌ ONNX Pipeline failed: {e}")
        return None

def run_paddleocr_official(image_path, version="PP-OCRv4"):
    """Run official PaddleOCR"""
    print(f"\n🐼 Running PaddleOCR Official ({version})...")
    print("=" * 60)
    
    try:
        from paddleocr import PaddleOCR
        
        # Initialize PaddleOCR
        start_time = time.time()
        
        # Set version-specific parameters
        if version == "PP-OCRv4":
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch'  # Chinese + English
            )
        elif version == "PP-OCRv5":
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch'
            )
        else:
            # Default
            ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        init_time = time.time() - start_time
        print(f"   ✅ PaddleOCR initialized in {init_time:.3f}s")
        
        # Run OCR
        ocr_start = time.time()
        results = ocr.predict(image_path)
        total_time = time.time() - start_time
        ocr_time = time.time() - ocr_start
        
        # Process results (OCRResult object)
        if results and len(results) > 0:
            ocr_result = results[0]  # OCRResult object
            
            # Extract data from OCRResult (it's a dictionary-like object)
            if 'dt_polys' in ocr_result and 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                boxes = ocr_result['dt_polys']  # Detection boxes (numpy arrays)
                texts = ocr_result['rec_texts']  # Recognition texts
                scores = ocr_result['rec_scores']  # Recognition scores
                
                print(f"✅ PaddleOCR found {len(texts)} text regions")
                for i, (text, score) in enumerate(zip(texts, scores)):
                    print(f"   ✅ '{text}' (confidence: {score:.3f})")
                
                # Convert boxes to list format (similar to ONNX)
                converted_boxes = []
                for box in boxes:
                    # Convert numpy array to list of 8 values [x1,y1,x2,y2,x3,y3,x4,y4]
                    if hasattr(box, 'shape') and len(box) == 4 and len(box[0]) == 2:
                        flat_box = box.flatten().tolist()
                        converted_boxes.append(flat_box)
                    else:
                        print(f"   ⚠️  Unexpected box format: {box.shape if hasattr(box, 'shape') else type(box)}")
                        converted_boxes.append(box.tolist() if hasattr(box, 'tolist') else box)
                
                print(f"✅ PaddleOCR completed in {total_time:.3f}s")
                print(f"   Init: {init_time:.3f}s, OCR: {ocr_time:.3f}s")
                
                return {
                    'boxes': converted_boxes,
                    'texts': texts,
                    'scores': scores,
                    'full_results': ocr_result,
                    'init_time': init_time,
                    'ocr_time': ocr_time,
                    'total_time': total_time
                }
            else:
                print(f"   ⚠️  OCRResult missing expected keys")
                print(f"   Available keys: {list(ocr_result.keys()) if hasattr(ocr_result, 'keys') else 'No keys method'}")
                
        print(f"❌ PaddleOCR found no text regions")
        return {
            'boxes': [],
            'texts': [],
            'scores': [],
            'full_results': results,
            'init_time': init_time,
            'ocr_time': ocr_time, 
            'total_time': total_time
        }
        
    except Exception as e:
        print(f"❌ PaddleOCR failed: {e}")
        return None

def compare_results(onnx_result, paddle_result, image_name):
    """Compare results between ONNX and PaddleOCR"""
    print(f"\n📊 COMPARISON RESULTS for {image_name}")
    print("=" * 80)
    
    if onnx_result is None:
        print("❌ ONNX result is None - cannot compare")
        return
    
    if paddle_result is None:
        print("❌ PaddleOCR result is None - cannot compare")
        return
    
    # Basic statistics
    onnx_count = len(onnx_result['texts'])
    paddle_count = len(paddle_result['texts'])
    
    print(f"📈 DETECTION COUNT:")
    print(f"   ONNX Pipeline: {onnx_count} text regions")
    print(f"   PaddleOCR:     {paddle_count} text regions")
    print(f"   Difference:    {abs(onnx_count - paddle_count)}")
    
    # Performance comparison
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   ONNX Total Time:    {onnx_result['total_time']:.3f}s")
    print(f"   PaddleOCR Total:    {paddle_result['total_time']:.3f}s")
    if onnx_result['total_time'] > 0:
        print(f"   Speed Ratio:        {paddle_result['total_time']/onnx_result['total_time']:.2f}x")
    
    # Text comparison (if both found text)
    if onnx_count > 0 and paddle_count > 0:
        print(f"\n📝 TEXT COMPARISON:")
        print(f"   {'Index':<5} {'ONNX Text':<30} {'PaddleOCR Text':<30} {'Match':<8}")
        print("-" * 80)
        
        max_count = max(onnx_count, paddle_count)
        exact_matches = 0
        
        for i in range(max_count):
            onnx_text = onnx_result['texts'][i] if i < onnx_count else "N/A"
            paddle_text = paddle_result['texts'][i] if i < paddle_count else "N/A"
            
            # Simple text matching (could be improved)
            is_match = "✅" if onnx_text == paddle_text else "❌"
            if onnx_text == paddle_text and onnx_text != "N/A":
                exact_matches += 1
            
            # Truncate long texts for display
            onnx_display = (onnx_text[:27] + "...") if len(onnx_text) > 30 else onnx_text
            paddle_display = (paddle_text[:27] + "...") if len(paddle_text) > 30 else paddle_text
            
            print(f"   {i+1:<5} {onnx_display:<30} {paddle_display:<30} {is_match:<8}")
        
        # Accuracy metrics
        if max_count > 0:
            accuracy = exact_matches / min(onnx_count, paddle_count) * 100 if min(onnx_count, paddle_count) > 0 else 0
            print(f"\n🎯 ACCURACY METRICS:")
            print(f"   Exact Matches:      {exact_matches}/{min(onnx_count, paddle_count)}")
            print(f"   Text Accuracy:      {accuracy:.1f}%")
    
    # Score comparison
    if onnx_result['scores'] and paddle_result['scores']:
        onnx_avg_score = sum(onnx_result['scores']) / len(onnx_result['scores'])
        paddle_avg_score = sum(paddle_result['scores']) / len(paddle_result['scores'])
        
        print(f"\n📊 CONFIDENCE SCORES:")
        print(f"   ONNX Avg Score:     {onnx_avg_score:.3f}")
        print(f"   PaddleOCR Avg:      {paddle_avg_score:.3f}")
    
    print("=" * 80)

def save_comparison_report(onnx_result, paddle_result, image_name, output_dir="output"):
    """Save detailed comparison report to JSON"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"comparison_report_{timestamp}.json")
    
    report = {
        "image_name": image_name,
        "timestamp": timestamp,
        "onnx_result": onnx_result,
        "paddle_result": paddle_result,
        "comparison": {
            "onnx_text_count": len(onnx_result['texts']) if onnx_result else 0,
            "paddle_text_count": len(paddle_result['texts']) if paddle_result else 0,
            "onnx_total_time": onnx_result['total_time'] if onnx_result else 0,
            "paddle_total_time": paddle_result['total_time'] if paddle_result else 0
        }
    }
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                return obj
            
            # Deep convert numpy objects
            import json
            json_string = json.dumps(report, ensure_ascii=False, indent=2, default=convert_numpy)
            f.write(json_string)
        print(f"📄 Comparison report saved: {report_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {e}")

def choose_test_image():
    """Choose test image interactively"""
    test_dir = "test"
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory '{test_dir}' not found!")
        return None
    
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"❌ No image files found in '{test_dir}' directory!")
        return None
    
    print(f"\n📸 Available Test Images:")
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_dir, img_file)
        try:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            file_size = os.path.getsize(img_path) / 1024  # KB
            print(f"{i}. {img_file} ({w}x{h}, {file_size:.1f}KB)")
        except:
            print(f"{i}. {img_file} (unable to read)")
    
    while True:
        try:
            choice = input(f"\nEnter your choice (1-{len(image_files)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(image_files):
                return os.path.join(test_dir, image_files[choice_idx])
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(image_files)}")
        except ValueError:
            print("❌ Please enter a number")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return None

def main():
    """Main comparison function"""
    print("🔍 PP-OCR ONNX vs PaddleOCR Official Comparison")
    print("=" * 80)
    
    # Check if PaddleOCR is available
    if not install_paddleocr_if_needed():
        print("❌ Cannot proceed without PaddleOCR")
        return
    
    # Choose test image
    image_path = choose_test_image()
    if not image_path:
        return
    
    image_name = os.path.basename(image_path)
    print(f"\n🖼️  Selected image: {image_name}")
    
    # Choose model version for ONNX
    print(f"\n🔧 Choose ONNX Model Version:")
    print("1. PP-OCRv4 (det_model_v4.onnx, rec_model_v4.onnx)")
    print("2. PP-OCRv5 (det_model_v5.onnx, rec_model_v5.onnx)")
    
    while True:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice == "1":
                onnx_version = "v4"
                paddle_version = "PP-OCRv4"
                break
            elif choice == "2":
                onnx_version = "v5"
                paddle_version = "PP-OCRv5"
                break
            else:
                print("❌ Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return
    
    print(f"\n🚀 Starting comparison...")
    print(f"   ONNX Version: {onnx_version}")
    print(f"   PaddleOCR Version: {paddle_version}")
    
    # Run both pipelines
    onnx_result = run_onnx_pipeline(image_path, onnx_version)
    paddle_result = run_paddleocr_official(image_path, paddle_version)
    
    # Compare results
    compare_results(onnx_result, paddle_result, image_name)
    
    # Save detailed report
    save_comparison_report(onnx_result, paddle_result, image_name)
    
    print(f"\n✅ Comparison completed!")

if __name__ == "__main__":
    main()
