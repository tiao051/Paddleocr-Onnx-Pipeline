"""
Complete PP-OCRv5 ONNX Pipeline
Integrates Detection → Cropping → Recognition → Final Text Output
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple, Dict, Optional

# Import custom modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from det.postprocessing_onnx import DBPostProcessONNX
from rec.rec_preprocessing_onnx import preprocess_ppocrv5, preprocess_ppocrv5_batch
from rec.rec_postprocessing_onnx import CTCLabelDecodeONNX
from utils.crop import get_rotate_crop_image


class PP_OCRv5_Pipeline:
    """
    Complete PP-OCRv5 ONNX Pipeline
    
    Workflow:
    1. Load ONNX models (detection + recognition)
    2. Preprocess image for detection
    3. Run detection → get text boxes
    4. Crop text regions from boxes
    5. Preprocess crops for recognition
    6. Run recognition → get text
    7. Return final results
    """
    
    def __init__(self, 
                 det_model_path: str,
                 rec_model_path: str,
                 det_target_size: int = 640,
                 rec_target_height: int = 48,
                 character_dict_path: str = None,
                 use_gpu: bool = False):
        """
        Initialize OCR pipeline
        
        Args:
            det_model_path: Path to detection ONNX model
            rec_model_path: Path to recognition ONNX model  
            det_target_size: Target size for detection preprocessing
            rec_target_height: Target height for recognition
            character_dict_path: Path to character dictionary
            use_gpu: Whether to use GPU inference
        """
        self.det_target_size = det_target_size
        self.rec_target_height = rec_target_height
        
        # Setup providers
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        
        # Load ONNX models
        print("Loading detection model...")
        self.det_session = ort.InferenceSession(det_model_path, providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        
        print("Loading recognition model...")  
        self.rec_session = ort.InferenceSession(rec_model_path, providers=providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        
        # Initialize postprocessors
        self.det_postprocessor = DBPostProcessONNX(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
            score_mode="fast",
            box_type='quad'
        )
        
        self.rec_postprocessor = CTCLabelDecodeONNX(
            character_dict_path=character_dict_path,
            use_space_char=True
        )
        
        print("✅ PP-OCRv5 Pipeline initialized successfully!")
    
    def preprocess_detection(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess image for detection
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            (input_tensor, scale_ratios)
        """
        original_h, original_w = image.shape[:2]
        ratio = self.det_target_size / max(original_h, original_w)
        new_h = int(original_h * ratio)
        new_w = int(original_w * ratio)
        
        # Round to multiple of 32
        new_h = max(32, int(np.round(new_h / 32)) * 32)
        new_w = max(32, int(np.round(new_w / 32)) * 32)
        
        resized_img = cv2.resize(image, (new_w, new_h))
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = resized_img.astype(np.float32) / 255.0
        normalized = (norm_img - mean) / std
        
        # HWC -> CHW
        chw = normalized.transpose(2, 0, 1)
        input_tensor = chw[np.newaxis, ...].astype(np.float32)
        
        # Scale ratios for postprocessing
        scale_h = new_h / original_h
        scale_w = new_w / original_w
        
        return input_tensor, (scale_h, scale_w)
    
    def run_detection(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run text detection on image
        
        Args:
            image: Input image
            
        Returns:
            (detected_boxes, confidence_scores)
        """
        # Preprocess
        input_tensor, (scale_h, scale_w) = self.preprocess_detection(image)
        
        # Run ONNX inference
        det_output = self.det_session.run([None], {self.det_input_name: input_tensor})[0]
        
        # Postprocess
        shape_info = [scale_h, scale_w]
        boxes, scores = self.det_postprocessor(det_output, shape_info)
        
        return boxes, scores
    
    def crop_text_regions(self, image: np.ndarray, boxes: List[np.ndarray]) -> List[np.ndarray]:
        """
        Crop text regions from detected boxes
        
        Args:
            image: Original image
            boxes: List of text boxes (4 points each)
            
        Returns:
            List of cropped text images
        """
        crops = []
        
        for i, box in enumerate(boxes):
            try:
                # Convert to proper format
                if isinstance(box, list):
                    box = np.array(box)
                if box.shape == (8,):
                    box = box.reshape(4, 2)
                
                # Crop using perspective transform
                cropped = get_rotate_crop_image(image, box)
                
                # Resize to target height
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    height, width = cropped.shape[:2]
                    ratio = self.rec_target_height / height
                    target_width = max(int(width * ratio), self.rec_target_height)
                    
                    resized = cv2.resize(cropped, (target_width, self.rec_target_height))
                    crops.append(resized)
                else:
                    print(f"Warning: Empty crop for box {i}")
                    
            except Exception as e:
                print(f"Error cropping box {i}: {e}")
                continue
        
        return crops
    
    def run_recognition(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Run text recognition on cropped images
        
        Args:
            crops: List of cropped text images
            
        Returns:
            List of (text, confidence) tuples
        """
        if not crops:
            return []
        
        # Batch preprocessing
        try:
            batch_tensor, indices = preprocess_ppocrv5_batch(crops)
        except:
            # Fallback to individual processing
            results = []
            for crop in crops:
                try:
                    input_tensor = preprocess_ppocrv5(crop)
                    rec_output = self.rec_session.run([None], {self.rec_input_name: input_tensor})[0]
                    decoded = self.rec_postprocessor(rec_output)
                    results.append(decoded[0] if decoded else ("", 0.0))
                except Exception as e:
                    print(f"Error in individual recognition: {e}")
                    results.append(("", 0.0))
            return results
        
        # Batch inference
        rec_output = self.rec_session.run([None], {self.rec_input_name: batch_tensor})[0]
        
        # Decode results
        decoded_results = self.rec_postprocessor(rec_output)
        
        # Restore original order
        results = [""] * len(crops)
        for i, original_idx in enumerate(indices):
            if i < len(decoded_results):
                results[original_idx] = decoded_results[i]
            else:
                results[original_idx] = ("", 0.0)
        
        return results
    
    def __call__(self, image: np.ndarray) -> List[Dict]:
        """
        Complete OCR pipeline
        
        Args:
            image: Input image
            
        Returns:
            List of detection results with text and confidence
        """
        # Step 1: Text Detection
        print("🔍 Running text detection...")
        boxes, det_scores = self.run_detection(image)
        
        if not boxes:
            print("⚠️ No text boxes detected")
            return []
        
        print(f"📦 Found {len(boxes)} text boxes")
        
        # Step 2: Crop text regions  
        print("✂️ Cropping text regions...")
        crops = self.crop_text_regions(image, boxes)
        
        if not crops:
            print("⚠️ No valid crops extracted")
            return []
        
        print(f"🖼️ Extracted {len(crops)} valid crops")
        
        # Step 3: Text Recognition
        print("🔤 Running text recognition...")
        rec_results = self.run_recognition(crops)
        
        # Step 4: Combine results
        final_results = []
        for i, (box, det_score) in enumerate(zip(boxes, det_scores)):
            if i < len(rec_results):
                text, rec_conf = rec_results[i]
                result = {
                    'text': text,
                    'confidence': rec_conf,
                    'det_score': det_score,
                    'box': box.tolist() if isinstance(box, np.ndarray) else box
                }
                final_results.append(result)
        
        print(f"✅ OCR completed: {len(final_results)} results")
        return final_results
    
    def visualize_results(self, image: np.ndarray, results: List[Dict], 
                         save_path: str = None) -> np.ndarray:
        """
        Visualize OCR results on image
        
        Args:
            image: Original image
            results: OCR results from __call__
            save_path: Path to save result image
            
        Returns:
            Annotated image
        """
        vis_img = image.copy()
        
        for i, result in enumerate(results):
            box = np.array(result['box'])
            text = result['text']
            confidence = result['confidence']
            
            # Reshape box if needed
            if box.shape == (8,):
                box = box.reshape(4, 2)
            
            # Draw bounding box
            pts = box.astype(np.int32)
            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
            
            # Put text and confidence
            x, y = pts[0]
            label = f"{text} ({confidence:.2f})"
            cv2.putText(vis_img, label, (x, max(10, y - 5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, vis_img)
            print(f"📸 Result saved to: {save_path}")
        
        return vis_img


def main():
    """Demo of complete OCR pipeline"""
    print("🚀 PP-OCRv5 ONNX Pipeline Demo")
    print("=" * 50)
    
    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    det_model = os.path.join(project_root, "models", "det_model.onnx")
    rec_model = os.path.join(project_root, "models", "rec_model.onnx")
    test_image = os.path.join(project_root, "test", "test.jpg")
    
    # Check if models exist
    if not os.path.exists(det_model):
        print(f"❌ Detection model not found: {det_model}")
        return
    if not os.path.exists(rec_model):
        print(f"❌ Recognition model not found: {rec_model}")
        return
    
    # Initialize pipeline
    try:
        ocr = PP_OCRv5_Pipeline(
            det_model_path=det_model,
            rec_model_path=rec_model,
            det_target_size=640,
            rec_target_height=48,
            use_gpu=False
        )
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Load test image
    if os.path.exists(test_image):
        image = cv2.imread(test_image)
    else:
        print("📸 Creating test image...")
        image = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(image, "Hello World", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "PP-OCRv5 Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "ONNX Pipeline", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print(f"📷 Processing image: {image.shape}")
    
    # Run OCR
    try:
        results = ocr(image)
        
        # Display results
        print("\n📋 OCR Results:")
        print("-" * 50)
        for i, result in enumerate(results):
            print(f"Text {i+1}: '{result['text']}'")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Det Score: {result['det_score']:.3f}")
            print(f"  Box: {result['box']}")
            print()
        
        # Visualize
        output_path = os.path.join(project_root, "output", "ocr_result.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vis_img = ocr.visualize_results(image, results, output_path)
        
        print(f"✅ Pipeline completed successfully!")
        print(f"📊 Processed {len(results)} text regions")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
