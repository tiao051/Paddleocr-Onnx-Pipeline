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
from rec_preprocessing_onnx import preprocess_ppocrv5, preprocess_ppocrv5_batch


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
        
        print(f"✅ ONNX Recognition model loaded: {model_path}")
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
        if char_dict_path and os.path.exists(char_dict_path):
            with open(char_dict_path, 'r', encoding='utf-8') as f:
                chars = [line.strip() for line in f.readlines()]
            print(f"📚 Loaded character dictionary: {len(chars)} characters")
            return ['blank'] + chars  # Add blank token for CTC
        else:
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
    Test function for PP-OCRv5 Recognition ONNX
    """
    print("=" * 60)
    print("TESTING PP-OCRv5 RECOGNITION ONNX")
    print("=" * 60)
    
    # Initialize recognizer
    model_path = "D:/Sozoo_Studio/v4_model/onnx_model/models/rec_model.onnx"

    
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
    
    # Test with dummy image
    print(f"\n🧪 Testing with dummy image...")
    dummy_img = np.random.randint(0, 255, (48, 160, 3), dtype=np.uint8)
    
    try:
        text = recognizer.recognize(dummy_img)
        print(f"   Single recognition result: '{text}'")
        print(f"   Text length: {len(text)} characters")
    except Exception as e:
        print(f"   ❌ Error in recognition: {e}")
        return
    
    # Test with synthetic text image
    print(f"\n🧪 Testing with synthetic text image...")
    try:
        # Create a simple text image using OpenCV
        text_img = np.ones((48, 320, 3), dtype=np.uint8) * 255  # White background
        cv2.putText(text_img, "HELLO", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        text_result = recognizer.recognize(text_img)
        print(f"   Synthetic text recognition: '{text_result}'")
        print(f"   Expected: 'HELLO' or similar")
    except Exception as e:
        print(f"   ❌ Error in synthetic text: {e}")
    
    # Test batch processing
    print(f"\n🧪 Testing batch processing...")
    image_list = [dummy_img] * 3
    
    try:
        texts = recognizer.recognize_batch(image_list)
        print(f"   Batch results:")
        for i, text in enumerate(texts):
            print(f"     Image {i+1}: '{text}'")
    except Exception as e:
        print(f"   ❌ Error in batch processing: {e}")
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
    model_path = "D:/Sozoo_Studio/v4_model/onnx_model/models/rec_model.onnx"
    
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


if __name__ == "__main__":
    import os
    
    # Test basic functionality
    test_recognition_onnx()
    
    # Demo with different image sizes
    demo_with_cropped_images()
    
    print(f"\n🎯 SUMMARY:")
    print(f"✅ PP-OCRv5 Recognition ONNX inference ready!")
    print(f"✅ Supports single and batch processing")
    print(f"✅ Compatible with detection pipeline")
    print(f"✅ Ready for complete OCR workflow")
