#!/usr/bin/env python3
"""
Quick test to understand PaddleOCR output format
"""

from paddleocr import PaddleOCR
import time

def test_paddleocr_format():
    """Test PaddleOCR output structure"""
    print("🔍 Testing PaddleOCR Output Format")
    print("=" * 50)
    
    # Initialize PaddleOCR 
    print("Initializing PaddleOCR...")
    start_time = time.time()
    
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
        
        init_time = time.time() - start_time
        print(f"✅ PaddleOCR initialized in {init_time:.3f}s")
        
        # Test image
        image_path = "test/test.jpg"
        
        print(f"\n🖼️  Testing with: {image_path}")
        
        # Run OCR
        print("Running OCR...")
        ocr_start = time.time()
        results = ocr.predict(image_path)
        ocr_time = time.time() - ocr_start
        
        print(f"✅ OCR completed in {ocr_time:.3f}s")
        
        # Analyze structure
        print(f"\n🔍 RESULT STRUCTURE ANALYSIS:")
        print(f"Type: {type(results)}")
        print(f"Length: {len(results) if results else 'None'}")
        
        if results:
            print(f"First element type: {type(results[0])}")
            print(f"First element length: {len(results[0]) if results[0] else 'None'}")
            
        if results:
            print(f"First element type: {type(results[0])}")
            
            # Handle OCRResult object
            ocr_result = results[0]
            print(f"OCR Result: {ocr_result}")
            print(f"OCR Result type: {type(ocr_result)}")
            
            # Try to access attributes - check correct attribute names
            try:
                print(f"Dir (filtered): {[attr for attr in dir(ocr_result) if not attr.startswith('_')]}")
                
                # Check all possible attribute names
                attrs_to_check = ['dt_polys', 'rec_texts', 'rec_scores', 'rec_text', 'rec_score', 
                                'boxes', 'texts', 'scores', 'detection_result', 'recognition_result']
                
                for attr in attrs_to_check:
                    if hasattr(ocr_result, attr):
                        value = getattr(ocr_result, attr)
                        print(f"✅ Found {attr}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'No length'}")
                        if attr in ['rec_texts', 'texts'] and hasattr(value, '__len__') and len(value) > 0:
                            print(f"   First few texts: {value[:3]}")
                        elif attr in ['dt_polys', 'boxes'] and hasattr(value, '__len__') and len(value) > 0:
                            print(f"   First box shape: {value[0].shape if hasattr(value[0], 'shape') else type(value[0])}")
                
                # Try dictionary access if it's dict-like
                if hasattr(ocr_result, 'keys'):
                    print(f"📋 Dictionary keys: {list(ocr_result.keys())}")
                    
            except Exception as e:
                print(f"Error accessing attributes: {e}")
                
                # Try iteration
                try:
                    print(f"Trying to iterate...")
                    for i, item in enumerate(ocr_result):
                        print(f"Item {i}: {item}")
                        if i >= 2:  # Limit output
                            break
                except Exception as e2:
                    print(f"Cannot iterate: {e2}")
                    
                    # Try string conversion
                    print(f"String representation: {str(ocr_result)}")
                    
        print(f"\n📊 SUMMARY:")
        print(f"Total results: {len(results) if results else 0}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_paddleocr_format()
