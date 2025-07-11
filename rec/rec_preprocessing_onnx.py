"""
PP-OCRv5 Mobile Recognition Preprocessing for ONNX

This module provides preprocessing specifically optimized for PP-OCRv5 mobile recognition model.

Key specifications from your config:
- Model: PP-OCRv5_mobile_rec
- Algorithm: SVTR_LCNet with PPLCNetV3 backbone
- Input shape: [3, 48, 320] (C, H, W)
- Max text length: 25 characters
- Original training: BGR input mode
- Normalization: PaddleOCR style (mean=0.5, std=0.5) → [-1, 1] range

Usage:
    # For PP-OCRv5 mobile (recommended)
    preprocessed = preprocess_ppocrv5_mobile(image_path_or_array)
    
    # Or with custom parameters
    preprocessed = preprocess_rec_image(image, image_shape=(3, 48, 320))
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple


def resize_norm_img(img, 
                   image_shape=(3, 32, 100), 
                   padding=False, 
                   interpolation=cv2.INTER_LINEAR,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   pad_mode='right'):
    """
    Resize and normalize image for recognition model
    
    Args:
        img: Input image (H, W, C) in RGB format
        image_shape: Target shape (C, H, W)
        padding: Whether to keep aspect ratio with padding
        interpolation: Interpolation method
        mean: Normalization mean values
        std: Normalization std values  
        pad_mode: 'right', 'center', 'left' - where to place resized image
    """
    imgC, imgH, imgW = image_shape
    h, w = img.shape[:2]
    
    # Validate input
    if len(img.shape) != 3 or img.shape[2] != imgC:
        raise ValueError(f"Expected image shape (H, W, {imgC}), got {img.shape}")
    
    # Calculate resize dimensions
    if padding:
        ratio = w / float(h)
        resized_w = min(int(np.ceil(imgH * ratio)), imgW)
    else:
        resized_w = imgW

    # Resize image
    img_resized = cv2.resize(img, (resized_w, imgH), interpolation=interpolation)
    
    # Create padded image
    resized_img = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
    
    # Place resized image based on pad_mode
    if pad_mode == 'center':
        start_x = (imgW - resized_w) // 2
        resized_img[:, start_x:start_x + resized_w, :] = img_resized
    elif pad_mode == 'left':
        resized_img[:, :resized_w, :] = img_resized
    else:  # 'right' - default
        resized_img[:, 0:resized_w, :] = img_resized

    # Normalize to [0, 1]
    resized_img = resized_img.astype('float32') / 255.0
    
    # Apply mean and std normalization
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    if len(mean) == 1:
        # Grayscale normalization
        resized_img = (resized_img - mean[0]) / std[0]
    else:
        # RGB normalization
        resized_img = (resized_img - mean) / std
    
    # Transpose HWC -> CHW
    resized_img = resized_img.transpose(2, 0, 1)
    return resized_img


def preprocess_rec_image(image_input: Union[str, np.ndarray], 
                        image_shape=(3, 48, 320),  # PP-OCRv5 default
                        padding=True,               # Keep aspect ratio
                        mean=[0.5, 0.5, 0.5],      # PaddleOCR style
                        std=[0.5, 0.5, 0.5],       # Range [-1, 1]
                        pad_mode='right'):
    """
    Complete preprocessing for PP-OCRv5 recognition model
    
    Args:
        image_input: Image path or numpy array
        image_shape: (C, H, W) — PP-OCRv5 uses (3, 48, 320)
        padding: True for dynamic width with aspect ratio (recommended)
        mean: Normalization mean values [0.5, 0.5, 0.5] for PaddleOCR
        std: Normalization std values [0.5, 0.5, 0.5] for range [-1, 1]
        pad_mode: Padding placement mode
        
    Returns:
        np.ndarray: Preprocessed tensor (1, C, H, W) ready for PP-OCRv5
    """
    # Load image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Cannot read image from {image_input}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
        # Assume input is RGB if 3 channels
        if len(img.shape) == 3 and img.shape[2] == 3:
            pass  # Already RGB
        elif len(img.shape) == 2:
            # Grayscale to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")
    
    # Handle grayscale input
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Preprocessing
    norm_img = resize_norm_img(img, 
                              image_shape=image_shape, 
                              padding=padding,
                              mean=mean,
                              std=std,
                              pad_mode=pad_mode)
    
    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    input_tensor = np.expand_dims(norm_img, axis=0)
    return input_tensor


def preprocess_rec_image_simple(image_input, image_shape=(3, 48, 320), padding=True):
    """
    Simple preprocessing optimized for PP-OCRv5 mobile recognition
    Default settings match PP-OCRv5 training configuration
    """
    return preprocess_rec_image(
        image_input=image_input,
        image_shape=image_shape,
        padding=padding,
        mean=[0.5, 0.5, 0.5],  # PP-OCRv5 standard
        std=[0.5, 0.5, 0.5],   # Results in [-1, 1] range
        pad_mode='right'
    )


def preprocess_ppocrv5_mobile(image_input):
    """
    Specialized preprocessing function for PP-OCRv5 mobile recognition
    Uses exact settings from your config.yml
    """
    config = get_rec_config('ppocrv5_mobile')
    return preprocess_rec_image(image_input, **config)


# Additional utility functions
def preprocess_rec_batch(image_list, **kwargs):
    """
    Batch preprocessing for multiple images
    """
    batch_tensors = []
    for img in image_list:
        tensor = preprocess_rec_image(img, **kwargs)
        batch_tensors.append(tensor)
    return np.concatenate(batch_tensors, axis=0)


def get_rec_config(model_type='ppocrv5_mobile'):
    """
    Get preprocessing config for different model types
    """
    configs = {
        # PP-OCRv5 Mobile Recognition (Your model)
        'ppocrv5_mobile': {
            'image_shape': (3, 48, 320),  # From your config: d2s_train_image_shape
            'padding': True,
            'mean': [0.5, 0.5, 0.5],      # PaddleOCR standard
            'std': [0.5, 0.5, 0.5],       # Range [-1, 1]
            'pad_mode': 'right',
            'max_text_length': 25
        },
        # Other PP-OCR versions for reference
        'ppocr_v2': {
            'image_shape': (3, 32, 320),
            'padding': True,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'pad_mode': 'right'
        },
        'ppocr_v3': {
            'image_shape': (3, 48, 320),
            'padding': True,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'pad_mode': 'right'
        },
        'ppocr_v4': {
            'image_shape': (3, 48, 320),
            'padding': True,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'pad_mode': 'right'
        },
        # Generic configs
        'crnn': {
            'image_shape': (3, 32, 100),
            'padding': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'pad_mode': 'right'
        },
        'svtr': {
            'image_shape': (3, 64, 256),
            'padding': True,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'pad_mode': 'right'
        },
        'trocr': {
            'image_shape': (3, 224, 224),
            'padding': False,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'pad_mode': 'center'
        }
    }
    return configs.get(model_type, configs['ppocrv5_mobile'])


# Example usage
if __name__ == "__main__":
    # Test with dummy image for PP-OCRv5
    dummy_img = np.random.randint(0, 255, (48, 160, 3), dtype=np.uint8)
    
    print("=== PP-OCRv5 Mobile Recognition Preprocessing Test ===")
    
    # PP-OCRv5 mobile preprocessing (recommended)
    result_v5 = preprocess_ppocrv5_mobile(dummy_img)
    print(f"PP-OCRv5 Mobile: {result_v5.shape}, range: [{result_v5.min():.3f}, {result_v5.max():.3f}]")
    
    # Simple preprocessing
    result_simple = preprocess_rec_image_simple(dummy_img)
    print(f"Simple (v5 settings): {result_simple.shape}, range: [{result_simple.min():.3f}, {result_simple.max():.3f}]")
    
    # Test different model configs
    print("\n=== Different Model Configs ===")
    for model_type in ['ppocrv5_mobile', 'ppocr_v4', 'ppocr_v3', 'ppocr_v2']:
        config = get_rec_config(model_type)
        try:
            # Resize dummy image to match expected input
            h, w = config['image_shape'][1], config['image_shape'][2]
            test_img = cv2.resize(dummy_img, (min(w, 160), h))
            result = preprocess_rec_image(test_img, **config)
            print(f"{model_type}: {result.shape}")
        except Exception as e:
            print(f"{model_type}: Error - {e}")
    
    print("\n=== Ready for PP-OCRv5 ONNX Inference! ===")


