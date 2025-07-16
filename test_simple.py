#!/usr/bin/env python3
"""
Simple comparison test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from paddle_vs_onnx_comparison import run_onnx_pipeline, run_paddleocr_official, compare_results

def test_comparison():
    """Test comparison with hardcoded inputs"""
    print("🔍 Testing PP-OCR ONNX vs PaddleOCR Official Comparison")
    print("=" * 80)
    
    image_path = "test/test.jpg"
    onnx_version = "v4"
    paddle_version = "PP-OCRv4"
    
    print(f"📸 Image: {image_path}")
    print(f"🔧 ONNX Version: {onnx_version}")
    print(f"🐼 PaddleOCR Version: {paddle_version}")
    print()
    
    # Run ONNX pipeline
    print("🚀 Running ONNX Pipeline...")
    onnx_result = run_onnx_pipeline(image_path, onnx_version)
    
    # Run PaddleOCR
    print("\n🐼 Running PaddleOCR...")
    paddle_result = run_paddleocr_official(image_path, paddle_version)
    
    # Compare results
    print("\n📊 Comparing Results...")
    comparison = compare_results(onnx_result, paddle_result, image_path)
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_comparison()
