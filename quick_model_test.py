"""
Quick test function to check if ONNX models in models/ folder are valid PP-OCRv5 models
"""
import os
import onnxruntime as ort
import numpy as np


def quick_test_models():
    """
    Quick test function to validate 2 PP-OCRv5 ONNX models
    Returns True if both models are valid, False otherwise
    """
    print("🔍 Quick PP-OCRv5 Model Test")
    print("=" * 40)
    
    models_to_check = {
        "det_model.onnx": {
            "type": "Detection",
            "expected_shape": [1, 3, 640, 640]
        },
        "rec_model.onnx": {
            "type": "Recognition", 
            "expected_shape": [1, 3, 48, 320]
        }
    }
    
    all_valid = True
    
    for model_name, config in models_to_check.items():
        model_path = os.path.join("models", model_name)
        
        print(f"\n📋 Testing {model_name} ({config['type']})...")
        
        # Check 1: File exists
        if not os.path.exists(model_path):
            print(f"   ❌ File not found: {model_path}")
            all_valid = False
            continue
        
        # Check 2: File size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        if file_size < 1:
            print(f"   ❌ File too small: {file_size:.1f}MB")
            all_valid = False
            continue
        else:
            print(f"   ✅ File size OK: {file_size:.1f}MB")
        
        try:
            # Check 3: Load ONNX model
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            
            # Check 4: Input shape
            input_info = session.get_inputs()[0]
            actual_shape = input_info.shape
            expected_shape = config["expected_shape"]
            
            # Handle dynamic dimensions
            shape_ok = True
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if isinstance(actual, str):  # Dynamic dimension
                    continue  # Skip dynamic dims
                elif expected != actual:
                    shape_ok = False
                    break
            
            if shape_ok:
                print(f"   ✅ Input shape valid: {actual_shape}")
            else:
                print(f"   ❌ Shape mismatch: expected {expected_shape}, got {actual_shape}")
                all_valid = False
                continue
            
            # Check 5: Quick inference test
            test_shape = []
            for dim in actual_shape:
                if isinstance(dim, str):
                    test_shape.append(1)  # Use 1 for dynamic dims
                else:
                    test_shape.append(dim)
            
            dummy_input = np.random.rand(*test_shape).astype(np.float32)
            outputs = session.run(None, {input_info.name: dummy_input})
            
            print(f"   ✅ Inference test passed")
            print(f"   📊 Outputs: {len(outputs)} tensors")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            all_valid = False
    
    # Final result
    print(f"\n{'='*40}")
    if all_valid:
        print("🎉 All models are valid PP-OCRv5 ONNX models!")
        print("✅ Ready to run OCR pipeline")
    else:
        print("❌ Some models failed validation")
        print("🔧 Please check your model files")
    
    return all_valid


if __name__ == "__main__":
    # Run quick test
    result = quick_test_models()
    
    if result:
        print("\n🚀 You can now run: python main.py")
    else:
        print("\n⚠️  Fix model issues before running main pipeline")
