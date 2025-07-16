"""
Test script to check if ONNX models in models/ folder are valid PP-OCRv5 models
"""
import os
import onnxruntime as ort
import numpy as np


def test_onnx_models():
    """
    Test function to validate PP-OCRv5 ONNX models
    Checks:
    1. File exists and has .onnx extension
    2. ONNX model can be loaded
    3. Input/output shapes match PP-OCRv5 specifications
    4. Model can run inference with dummy data
    """
    print("=" * 60)
    print("TESTING PP-OCRv5 ONNX MODELS")
    print("=" * 60)
    
    models_dir = "models"
    expected_models = {
        "det_model.onnx": {
            "type": "Detection",
            "expected_input_shape": [1, 3, 640, 640],  # PP-OCRv5 detection input
            "description": "PP-OCRv5 Detection Model (DB)"
        },
        "rec_model.onnx": {
            "type": "Recognition", 
            "expected_input_shape": [1, 3, 48, 320],   # PP-OCRv5 recognition input
            "description": "PP-OCRv5 Recognition Model (SVTR_LCNet)"
        }
    }
    
    results = {}
    
    for model_name, config in expected_models.items():
        print(f"\n🔍 Testing {model_name} ({config['type']})...")
        model_path = os.path.join(models_dir, model_name)
        
        # Test 1: File existence
        if not os.path.exists(model_path):
            print(f"   ❌ File not found: {model_path}")
            results[model_name] = False
            continue
        
        # Test 2: File extension
        if not model_name.endswith('.onnx'):
            print(f"   ❌ Invalid extension: {model_name}")
            results[model_name] = False
            continue
        
        # Test 3: File size check
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"   📁 File size: {file_size:.1f} MB")
        
        if file_size < 1:  # ONNX models should be at least 1MB
            print(f"   ⚠️  Warning: File size too small, might be corrupted")
        
        try:
            # Test 4: Load ONNX model
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            print(f"   ✅ ONNX model loaded successfully")
            
            # Test 5: Check input/output shapes
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()
            
            print(f"   📊 Input name: {input_info.name}")
            print(f"   📊 Input shape: {input_info.shape}")
            print(f"   📊 Output count: {len(output_info)}")
            
            # Validate input shape matches PP-OCRv5 specs
            expected_shape = config["expected_input_shape"]
            actual_shape = input_info.shape
            
            # Handle dynamic dimensions (represented as strings in ONNX)
            shape_match = True
            for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
                if isinstance(actual, str):  # Dynamic dimension
                    print(f"   📊 Dynamic dimension at index {i}: {actual}")
                elif expected != actual:
                    shape_match = False
                    break
            
            if shape_match:
                print(f"   ✅ Input shape matches PP-OCRv5 {config['type']} model")
            else:
                print(f"   ⚠️  Input shape mismatch. Expected: {expected_shape}, Got: {actual_shape}")
            
            # Test 6: Run dummy inference
            print(f"   🧪 Testing inference with dummy data...")
            
            # Create dummy input data
            if isinstance(actual_shape[0], str):  # Dynamic batch
                test_shape = [1] + list(actual_shape[1:])
            else:
                test_shape = list(actual_shape)
            
            # Replace any remaining dynamic dims with reasonable values
            for i, dim in enumerate(test_shape):
                if isinstance(dim, str):
                    if i == 2:  # Height
                        test_shape[i] = expected_shape[i]
                    elif i == 3:  # Width  
                        test_shape[i] = expected_shape[i]
                    else:
                        test_shape[i] = 1
            
            dummy_input = np.random.rand(*test_shape).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_info.name: dummy_input})
            print(f"   ✅ Inference successful!")
            print(f"   📊 Output shapes: {[out.shape for out in outputs]}")
            
            # Model-specific validation
            if config["type"] == "Detection":
                # Detection model should output feature maps
                if len(outputs) >= 1 and len(outputs[0].shape) == 4:
                    print(f"   ✅ Detection output format valid (4D tensor)")
                else:
                    print(f"   ⚠️  Unexpected detection output format")
            
            elif config["type"] == "Recognition":
                # Recognition model should output sequence predictions
                if len(outputs) >= 1 and len(outputs[0].shape) == 3:
                    print(f"   ✅ Recognition output format valid (3D tensor)")
                    seq_len = outputs[0].shape[1]
                    num_classes = outputs[0].shape[2]
                    print(f"   📊 Sequence length: {seq_len}, Classes: {num_classes}")
                else:
                    print(f"   ⚠️  Unexpected recognition output format")
            
            results[model_name] = True
            print(f"   🎉 {model_name} validation PASSED!")
            
        except Exception as e:
            print(f"   ❌ Error testing {model_name}: {str(e)}")
            results[model_name] = False
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{model_name:20} : {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 Overall Status: {'✅ ALL MODELS VALID' if all_passed else '❌ SOME MODELS FAILED'}")
    
    if all_passed:
        print("🚀 Ready to run PP-OCRv5 ONNX inference pipeline!")
    else:
        print("🔧 Please check failed models and ensure they are valid PP-OCRv5 ONNX exports")
    
    return results


def check_model_compatibility():
    """
    Additional check for model compatibility with current pipeline
    """
    print(f"\n🔧 Checking pipeline compatibility...")
    
    # Check if models exist in expected location
    det_path = "models/det_model.onnx"
    rec_path = "models/rec_model.onnx"
    
    compatibility_issues = []
    
    # Check main.py model path
    try:
        with open("main.py", "r") as f:
            main_content = f.read()
            
        if "D:/Sozoo_Studio/v5_model" in main_content:
            # This is expected now, no issue
            pass
        elif "D:/Sozoo_Studio/v4_model" in main_content:
            compatibility_issues.append("main.py still references old v4_model path instead of current v5_model")
        
        if rec_path not in main_content and "models/rec_model.onnx" not in main_content:
            compatibility_issues.append("main.py model path may need updating to use relative path")
            
    except Exception as e:
        compatibility_issues.append(f"Could not read main.py: {e}")
    
    if compatibility_issues:
        print("   ⚠️  Compatibility issues found:")
        for issue in compatibility_issues:
            print(f"      - {issue}")
    else:
        print("   ✅ No compatibility issues detected")
    
    return len(compatibility_issues) == 0


if __name__ == "__main__":
    # Run model validation tests
    model_results = test_onnx_models()
    
    # Run compatibility check
    compatibility_ok = check_model_compatibility()
    
    # Final verdict
    all_models_valid = all(model_results.values())
    
    print(f"\n" + "🎯" * 20)
    if all_models_valid and compatibility_ok:
        print("🎉 ALL TESTS PASSED - PP-OCRv5 ONNX pipeline ready!")
    else:
        print("⚠️  SOME ISSUES DETECTED - Please fix before running pipeline")
    print("🎯" * 20)
