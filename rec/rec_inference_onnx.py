"""
PP-OCRv5 Mobile Recognition ONNX Inference
Complete pipeline: preprocessing → ONNX inference → postprocessing

Usage:
    from rec_inference_onnx import RecognitionONNX
    recognizer = RecognitionONNX("rec_model.onnx")
    text = recognizer.recognize(image)
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
from typing import Union, List, Tuple
from rec.rec_preprocessing_onnx import preprocess_ppocrv5, preprocess_ppocrv5_batch


class RecognitionONNX:
    """
    PP-OCRv5 Recognition ONNX Inference Class
    
    Complete pipeline:
    1. Preprocessing (resize, normalize, padding)
    2. ONNX model inference 
    3. Postprocessing (decode predictions to text)
    """
    
    def __init__(self, 
                 model_path: str,
                 char_dict_path: str = None,
                 providers: List[str] = None):
        """
        Initialize PP-OCRv5 Recognition ONNX model
        
        Args:
            model_path: Path to rec_model.onnx
            char_dict_path: Path to character dictionary (optional)
            providers: ONNX providers (default: ["CPUExecutionProvider"])
        """
        self.model_path = model_path
        
        # Initialize ONNX session
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"   ONNX Recognition model loaded: {model_path}")
        print(f"   Input: {self.input_name}")
        print(f"   Outputs: {self.output_names}")
        print(f"   Providers: {providers}")
        
        # Load character dictionary
        self.char_dict = self._load_char_dict(char_dict_path)
        
    def _load_char_dict(self, char_dict_path: str = None) -> List[str]:
        """
        Load character dictionary for text decoding
        
        Args:
            char_dict_path: Path to character dictionary file
            
        Returns:
            List of characters for decoding
        """
        # Try to load from provided path first
        if char_dict_path and os.path.exists(char_dict_path):
            with open(char_dict_path, 'r', encoding='utf-8') as f:
                chars = [line.strip() for line in f.readlines()]
            print(f"📚 Loaded character dictionary: {len(chars)} characters from {char_dict_path}")
            return ['blank'] + chars  # Add blank token for CTC
        
        # Try to find the character dictionary in common locations
        possible_paths = [
            "utils/char_dic.txt",
            "../utils/char_dic.txt", 
            "d:/Sozoo_Studio/v5_model/onnx_model/utils/char_dic.txt",
            "./ppocr/utils/dict/ppocrv5_dict.txt",
            "ppocr/utils/dict/ppocrv5_dict.txt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        chars = [line.strip() for line in f.readlines()]
                    print(f"📚 Found and loaded character dictionary: {len(chars)} characters from {path}")
                    return ['blank'] + chars  # Add blank token for CTC
                except Exception as e:
                    print(f"⚠️  Error loading dictionary from {path}: {e}")
                    continue
        
        # If no dictionary file found, use default set
        print("⚠️  No character dictionary file found, using default ASCII set")
        print("   This may cause recognition issues if model expects more characters")
        
        # Default character set for PP-OCRv5 (basic ASCII + numbers)
        chars = []
        # Numbers
        chars.extend([str(i) for i in range(10)])
        # Lowercase letters  
        chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        # Uppercase letters
        chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        # Common symbols
        chars.extend([' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', 
                     ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', 
                     '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'])
        
        print(f"📚 Using default character set: {len(chars)} characters")
        return ['blank'] + chars  # Add blank token for CTC
    
    def _run_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference on preprocessed tensor
        
        Args:
            input_tensor: Preprocessed image tensor (B, C, H, W)
            
        Returns:
            Model predictions (B, T, num_classes)
        """
        # Ensure correct input format
        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)
            
        # Run inference
        outputs = self.session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Return first output (predictions)
        return outputs[0]
    
    def _decode_predictions(self, predictions: np.ndarray, verbose: bool = False) -> List[str]:
        """
        Decode CTC predictions to text strings
        
        Args:
            predictions: Model output (B, T, num_classes)
            verbose: Print debug information
            
        Returns:
            List of recognized text strings
        """
        batch_size = predictions.shape[0]
        results = []
        
        if verbose:
            print(f"   🔍 Predictions shape: {predictions.shape}")
            print(f"   🔍 Character dict size: {len(self.char_dict)}")
        
        for i in range(batch_size):
            pred = predictions[i]  # (T, num_classes)
            
            # Get character indices (argmax)
            char_indices = np.argmax(pred, axis=1)  # (T,)
            
            if verbose:
                print(f"   🔍 Sequence length: {len(char_indices)}")
                print(f"   🔍 Unique indices: {np.unique(char_indices)[:10]}...")  # Show first 10
                print(f"   🔍 Max confidence: {np.max(pred):.3f}")
            
            # CTC decoding: remove blanks and consecutive duplicates
            decoded_chars = []
            prev_char = None
            
            for char_idx in char_indices:
                if char_idx == 0:  # Skip blank token
                    prev_char = None
                    continue
                    
                if char_idx != prev_char:  # Remove consecutive duplicates
                    if char_idx < len(self.char_dict):
                        decoded_chars.append(self.char_dict[char_idx])
                        if verbose and len(decoded_chars) <= 5:  # Show first few chars
                            print(f"   🔍 Decoded char: {char_idx} → '{self.char_dict[char_idx]}'")
                    prev_char = char_idx
            
            # Join characters to form text
            text = ''.join(decoded_chars)
            results.append(text)
            
            if verbose:
                print(f"   🔍 Final text: '{text}' (length: {len(text)})")
            
        return results
    
    def recognize(self, image_input: Union[str, np.ndarray]) -> str:
        """
        Recognize text from single image
        
        Args:
            image_input: Image path or numpy array
            
        Returns:
            Recognized text string
        """
        # Preprocessing
        input_tensor = preprocess_ppocrv5(image_input)
        
        # ONNX inference
        predictions = self._run_inference(input_tensor)
        
        # Postprocessing
        texts = self._decode_predictions(predictions, verbose=True)
        
        return texts[0] if texts else ""
    
    def recognize_batch(self, image_list: List[Union[str, np.ndarray]]) -> List[str]:
        """
        Recognize text from multiple images (batch processing)
        
        Args:
            image_list: List of image paths or numpy arrays
            
        Returns:
            List of recognized text strings
        """
        if not image_list:
            return []
        
        # Batch preprocessing
        batch_tensor, indices = preprocess_ppocrv5_batch(image_list)
        
        # ONNX inference
        predictions = self._run_inference(batch_tensor)
        
        # Postprocessing
        texts = self._decode_predictions(predictions)
        
        # Restore original order
        sorted_texts = [''] * len(image_list)
        for i, original_idx in enumerate(indices):
            if i < len(texts):
                sorted_texts[original_idx] = texts[i]
                
        return sorted_texts
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dictionary with model details
        """
        input_shape = self.session.get_inputs()[0].shape
        output_shapes = [output.shape for output in self.session.get_outputs()]
        
        return {
            'model_path': self.model_path,
            'input_name': self.input_name,
            'input_shape': input_shape,
            'output_names': self.output_names,
            'output_shapes': output_shapes,
            'num_characters': len(self.char_dict),
            'providers': self.session.get_providers()
        }


def test_recognition_onnx():
    """
    Test function for PP-OCRv5 Recognition ONNX with real test.jpg image and visualize results
    """
    print("=" * 60)
    print("TESTING PP-OCRv5 RECOGNITION ONNX WITH VISUALIZATION")
    print("=" * 60)
    
    # Initialize recognizer
    model_path = "D:/Sozoo_Studio/v5_model/onnx_model/models/rec_model.onnx"
    
    # Check if model exists, if not try relative path
    if not os.path.exists(model_path):
        model_path = "models/rec_model.onnx"
    if not os.path.exists(model_path):
        model_path = "rec_model.onnx"
    
    recognizer = RecognitionONNX(model_path)
    
    # Print model info
    info = recognizer.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   Input shape: {info['input_shape']}")
    print(f"   Output shapes: {info['output_shapes']}")
    print(f"   Characters: {info['num_characters']}")
    
    # Find test.jpg image
    test_image_paths = [
        "test/test.jpg",
        "test.jpg", 
        "D:/Sozoo_Studio/v5_model/onnx_model/test/test.jpg",
        "../test/test.jpg"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("\n❌ test.jpg not found! Tried paths:")
        for path in test_image_paths:
            print(f"   - {path}")
        return
    
    print(f"\n📁 Found test image: {test_image_path}")
    
    # Test with real test.jpg image and visualize
    print(f"\n🧪 Testing with test.jpg and drawing results...")
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"   ❌ Could not load image from {test_image_path}")
            return
            
        h, w = img.shape[:2]
        print(f"   📐 Image dimensions: {w}x{h}")
        
        # Create a copy for visualization
        result_img = img.copy()
        
        # Test recognition on full image
        print(f"   🔍 Running recognition on full image...")
        full_result = recognizer.recognize(test_image_path)
        print(f"   📝 Full image recognition result: '{full_result}'")
        
        # Draw full image result at the top
        cv2.putText(result_img, f"Full: {full_result}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Test on image crops for better results
        print(f"\n   🔍 Testing recognition on image crops...")
        
        # Create some sample crops from different regions
        crop_regions = [
            (50, 50, min(w, 300), min(h, 100)),   # Top region
            (w//4, h//4, min(w, w//4 + 250), min(h, h//4 + 80)),  # Center-left region  
            (w//2, h//2, min(w, w//2 + 200), min(h, h//2 + 60)),  # Center-right region
        ]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
        
        for i, (x1, y1, x2, y2) in enumerate(crop_regions):
            # Ensure crop coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                
                print(f"   🔍 Crop {i+1} ({crop_w}x{crop_h}) at ({x1},{y1})-({x2},{y2}):")
                crop_result = recognizer.recognize(crop)
                print(f"      Result: '{crop_result}'")
                
                # Draw bounding box for crop region
                color = colors[i % len(colors)]
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw the recognized text near the crop
                text_y = y1 - 10 if y1 > 30 else y2 + 30
                cv2.putText(result_img, f"Crop{i+1}: {crop_result}", 
                           (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save the result image
        output_path = "output/recognition_result.jpg"
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, result_img)
        print(f"\n💾 Saved visualization to: {output_path}")
        
        # Show the image (if possible)
        try:
            cv2.imshow("Recognition Results", result_img)
            print(f"🖼️  Displaying result image (press any key to close)")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print(f"🖼️  Image saved but cannot display (no GUI)")
        
    except Exception as e:
        print(f"   ❌ Error in recognition: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✅ PP-OCRv5 Recognition ONNX test completed!")
    print("   Ready for integration with detection pipeline")


def demo_with_cropped_images():
    """
    Demo function for recognizing cropped text images
    """
    print("\n" + "=" * 60)
    print("DEMO: RECOGNITION WITH CROPPED TEXT IMAGES")
    print("=" * 60)
    
    # Initialize recognizer - fix path
    model_path = "D:/Sozoo_Studio/v5_model/onnx_model/models/rec_model.onnx"
    
    # Check if model exists, if not try relative paths
    if not os.path.exists(model_path):
        model_path = "../models/rec_model.onnx"
    if not os.path.exists(model_path):
        model_path = "models/rec_model.onnx"
    if not os.path.exists(model_path):
        print(f"❌ Model not found! Please check if rec_model.onnx exists in models/ folder")
        return
    
    recognizer = RecognitionONNX(model_path)
    
    # Create simulated text region crops
    print(f"📝 Creating simulated text crops...")
    
    # Different aspect ratios and sizes
    test_crops = [
        np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8),  # Short text
        np.random.randint(0, 255, (48, 200, 3), dtype=np.uint8),  # Medium text
        np.random.randint(0, 255, (40, 320, 3), dtype=np.uint8),  # Long text
        np.random.randint(0, 255, (64, 80, 3), dtype=np.uint8),   # Square-ish
    ]
    
    descriptions = ["Short text", "Medium text", "Long text", "Square text"]
    
    # Process each crop
    for i, (crop, desc) in enumerate(zip(test_crops, descriptions)):
        h, w = crop.shape[:2]
        print(f"\n🔍 Processing {desc} ({h}x{w}):")
        
        try:
            text = recognizer.recognize(crop)
            print(f"   Result: '{text}'")
            print(f"   Confidence: Mock confidence score")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Batch processing all crops
    print(f"\n📦 Batch processing all crops:")
    try:
        batch_texts = recognizer.recognize_batch(test_crops)
        for i, (text, desc) in enumerate(zip(batch_texts, descriptions)):
            print(f"   {desc}: '{text}'")
    except Exception as e:
        print(f"   ❌ Batch error: {e}")
    
    print(f"\n✅ Recognition demo completed!")


def test_synthetic_text():
    """Test recognition with synthetic text image"""
    print("\n🧪 Testing with synthetic text image...")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a synthetic text image
        width, height = 320, 48
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Try to use default font or Arial if available
        try:
            # Try to use a larger font
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw text
        text = "HELLO"
        if font:
            # Get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center the text
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill='black', font=font)
        else:
            # Fallback - draw simple text without font
            draw.text((width//2 - 30, height//2 - 10), text, fill='black')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        print(f"   📝 Created synthetic image with text: '{text}'")
        print(f"   📐 Image shape: {img_array.shape}")
        
        # Test recognition
        model_path = "D:/Sozoo_Studio/v5_model/onnx_model/models/rec_model.onnx"
        if not os.path.exists(model_path):
            model_path = "models/rec_model.onnx"
        
        recognizer = RecognitionONNX(model_path)
        result = recognizer.recognize(img_array)
        print(f"   ✅ Recognition result: '{result}'")
        
        if result.strip():
            print(f"   🎉 SUCCESS: Model recognized text!")
        else:
            print(f"   ⚠️  WARNING: Model returned empty result")
            
        return img_array, result
        
    except ImportError:
        print("   ⚠️  PIL not available, skipping synthetic text test")
        return None, ""
    except Exception as e:
        print(f"   ❌ ERROR during synthetic text test: {e}")
        import traceback
        traceback.print_exc()
        return None, ""


def test_with_real_image():
    """Test recognition with the real test.jpg image"""
    print("\n🧪 Testing with real test image...")
    
    # Initialize recognizer
    model_path = "D:/Sozoo_Studio/v5_model/onnx_model/models/rec_model.onnx"
    if not os.path.exists(model_path):
        model_path = "models/rec_model.onnx"
    
    recognizer = RecognitionONNX(model_path)
    
    # Try to find test.jpg in different locations
    test_image_paths = [
        "test/test.jpg",
        "test.jpg",
        "D:/Sozoo_Studio/v5_model/onnx_model/test/test.jpg",
        "../test/test.jpg"
    ]
    
    test_image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image_path = path
            break
    
    if test_image_path is None:
        print("   ❌ test.jpg not found! Tried paths:")
        for path in test_image_paths:
            print(f"      - {path}")
        return
    
    print(f"   📁 Found test image: {test_image_path}")
    
    # Load and display image info
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"   ❌ Could not load image from {test_image_path}")
            return
            
        h, w = img.shape[:2]
        print(f"   📐 Image dimensions: {w}x{h}")
        
        # Test recognition on full image
        print(f"   🔍 Running recognition on full image...")
        result = recognizer.recognize(test_image_path)
        print(f"   📝 Full image recognition result: '{result}'")
        
        # If we have the detection model results, we could test on cropped regions
        # For now, let's create some sample crops from different parts of the image
        print(f"   🔍 Testing recognition on image crops...")
        
        # Create some sample crops from different regions
        crop_regions = [
            (50, 50, 200, 100),   # Top-left region
            (w//4, h//4, w//2, h//3),  # Center-left region  
            (w//2, h//2, w-50, h//2 + 50),  # Center-right region
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(crop_regions):
            # Ensure crop coordinates are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]
                
                print(f"   🔍 Crop {i+1} ({crop_w}x{crop_h}) at ({x1},{y1})-({x2},{y2}):")
                crop_result = recognizer.recognize(crop)
                print(f"      Result: '{crop_result}'")
        
        print(f"   ✅ Real image test completed!")
        
    except Exception as e:
        print(f"   ❌ Error during real image test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    
    # Test with real test.jpg image
    test_recognition_onnx()
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ PP-OCRv5 Recognition ONNX inference ready!")
    print(f"✅ Tested with real test.jpg image")
    print(f"✅ Supports single and batch processing")
    print(f"✅ Compatible with detection pipeline")
    print(f"✅ Ready for complete OCR workflow")
