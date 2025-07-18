"""
PP-OCRv5 Mobile Recognition Preprocessing for ONNX
Based on predict_rec.py from PaddleOCR

Simplified version ONLY for PP-OCRv5_rec.onnx model
"""

import cv2
import numpy as np
import math
from typing import Union, List


def resize_norm_img_ppocrv5(img, max_wh_ratio, rec_image_shape=(3, 48, 320)):
    imgC, imgH, imgW = rec_image_shape
    
    # Calculate dynamic width based on max_wh_ratio
    imgW = int((imgH * max_wh_ratio))
    
    h, w = img.shape[:2]
    ratio = w / float(h)
    
    # Calculate resized width while maintaining aspect ratio
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    
    # Resize image
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    
    # PaddleOCR normalization: /255, -0.5, /0.5 → [-1, 1]
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    
    # Zero padding (crucial for batch processing)
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    
    return padding_im


def preprocess_ppocrv5(image_input: Union[str, np.ndarray]):
    # Load and validate image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Cannot read image from {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")
    
    # Handle grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # PP-OCRv5 config
    rec_image_shape = (3, 48, 320)
    imgC, imgH, imgW = rec_image_shape
    
    # Calculate max_wh_ratio
    h, w = img.shape[:2]
    wh_ratio = w / float(h)
    max_wh_ratio = max(imgW / imgH, wh_ratio)  # At least 320/48 ≈ 6.67
    
    # Apply preprocessing
    norm_img = resize_norm_img_ppocrv5(img, max_wh_ratio, rec_image_shape)
    
    # Add batch dimension for ONNX
    input_tensor = np.expand_dims(norm_img, axis=0)
    return input_tensor


def preprocess_ppocrv5_batch(image_list: List[Union[str, np.ndarray]]):
    """
    Batch preprocessing for PP-OCRv5_rec.onnx model
    
    Args:
        image_list: List of image paths or numpy arrays
        
    Returns:
        tuple: (batch_tensor, indices) for restoring original order
    """
    img_num = len(image_list)
    
    # Load all images and calculate width ratios
    loaded_images = []
    width_list = []
    
    for image_input in image_list:
        # Load image
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Cannot read image from {image_input}")
        else:
            img = image_input.copy()
        
        # Handle grayscale
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        loaded_images.append(img)
        width_list.append(img.shape[1] / float(img.shape[0]))
    
    # Sort by width ratio for processing efficiency
    indices = np.argsort(np.array(width_list))
    
    # Calculate global max_wh_ratio for consistent width
    rec_image_shape = (3, 48, 320)
    imgC, imgH, imgW = rec_image_shape
    max_wh_ratio = imgW / imgH  # Start with 320/48 ≈ 6.67
    
    for i in range(img_num):
        h, w = loaded_images[i].shape[:2]
        wh_ratio = w / float(h)
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
    
    dynamic_width = int((imgH * max_wh_ratio))
    dynamic_rec_shape = (imgC, imgH, dynamic_width)
    
    # Process all images with same max_wh_ratio
    norm_img_batch = []
    for i in range(img_num):
        img_idx = indices[i]
        img = loaded_images[img_idx]
        
        norm_img = resize_norm_img_ppocrv5(img, max_wh_ratio, dynamic_rec_shape)
        norm_img_batch.append(norm_img[np.newaxis, :])
    
    # Concatenate batch
    batch_tensor = np.concatenate(norm_img_batch, axis=0)
    
    return batch_tensor, indices


# Example usage and testing
if __name__ == "__main__":
    # Test with dummy image
    dummy_img = np.random.randint(0, 255, (48, 160, 3), dtype=np.uint8)
    
    print("=== PP-OCRv5_rec.onnx Preprocessing Test ===")
    print("Based on predict_rec.py production code\n")
    
    # Single image preprocessing
    result = preprocess_ppocrv5(dummy_img)
    print(f"Single image: {result.shape}")
    print(f"Range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"Expected: (1, 3, 48, 320), range ≈ [-1, 1]\n")
    
    # Batch preprocessing
    image_list = [dummy_img] * 3
    batch_result, indices = preprocess_ppocrv5_batch(image_list)
    print(f"Batch processing: {batch_result.shape}")
    print(f"Sorted indices: {indices}")
    print(f"Expected: (3, 3, 48, 320)\n")
    
    # Test with different aspect ratios
    print("=== Different Aspect Ratio Tests ===")
    test_cases = [
        (32, 100, "Square-ish"),
        (48, 320, "Standard"),  
        (48, 640, "Wide"),
        (64, 32, "Tall")
    ]
    
    for h, w, desc in test_cases:
        test_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        result = preprocess_ppocrv5(test_img)
        print(f"{desc:12} ({h:2}x{w:3}): {result.shape} -> Dynamic width: {result.shape[3]}")
    
    print("\n✅ PP-OCRv5_rec.onnx preprocessing ready!")