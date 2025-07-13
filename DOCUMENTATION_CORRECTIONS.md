# Documentation Corrections Based on Actual Implementation

## 1. Model Specifications (Section 4) - CORRECTED WITH YAML CONFIGS

### Detection Model (PP-OCRv5_mobile_det):
- **Model Architecture**: DB (Differentiable Binarization) with PPLCNetV3 backbone (scale=0.75)
- **Actual Input**: `x` (input node name)
- **Actual Shape**: [1, 3, 640, 640] (from det_pp-ocrv5.yml: `d2s_train_image_shape: [3, 640, 640]`)
- **Actual Output**: Probability map [1, 1, H, W] where H,W are scaled versions of 640x640
- **Neck**: RSEFPN with 96 output channels and shortcut connections
- **Head**: DBHead with k=50 and fix_nan=True

### Recognition Model (PP-OCRv5_mobile_rec):
- **Model Architecture**: SVTR_LCNet with PPLCNetV3 backbone (scale=0.95)
- **Actual Input Shape**: [1, 3, 48, 320] (from rec_pp-ocrv5.yml: `d2s_train_image_shape: [3, 48, 320]`)
- **Dynamic Width**: Supports variable width based on max_wh_ratio, minimum 320px
- **Head**: MultiHead with CTCHead + NRTRHead
- **SVTR Neck**: dims=120, depth=2, hidden_dims=120, kernel_size=[1,3], use_guide=True
- **Max Text Length**: 25 characters
- **Normalization**: [-1, 1] range (not [0, 1])

### Classification Model:
- **Note**: Classification model is NOT actually used in this implementation
- The pipeline goes: Detection → Recognition (2-stage, not 3-stage)

## 2. Process Flow (Section 3) - Major Correction

### Actual Pipeline:
```
Input Image → Detection Preprocessing → Detection ONNX → Postprocessing → Crop Text Regions → Recognition Preprocessing → Recognition ONNX → CTC Decoding → Final Text
```

**Classification stage is NOT implemented** - this is a 2-stage pipeline, not 3-stage.

## 3. Preprocessing Details (Section 3.1) - CORRECTED WITH YAML EVIDENCE

### Detection Preprocessing (Based on det_pp-ocrv5.yml):
- **Target Size**: 640px (fixed from `d2s_train_image_shape: [3, 640, 640]`)
- **Rounding**: Round to nearest 32-pixel multiple for optimal processing
- **Normalization**: ImageNet normalization from YAML config:
  ```yaml
  NormalizeImage:
    scale: 1./255.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    order: hwc
  ```
- **Transform Pipeline**: DecodeImage → DetLabelEncode → DetResizeForTest → NormalizeImage → ToCHWImage

### Recognition Preprocessing (Based on rec_pp-ocrv5.yml):
- **Height**: Fixed 48px (from `d2s_train_image_shape: [3, 48, 320]`)
- **Width**: Dynamic based on aspect ratio, minimum 320px
- **Max Text Length**: 25 characters (from `max_text_length: 25`)
- **Character Dictionary**: `./ppocr/utils/dict/ppocrv5_dict.txt`
- **Space Character**: Enabled (`use_space_char: true`)
- **Normalization**: (pixel/255 - 0.5) / 0.5 → [-1, 1] range
- **Padding**: Zero-padding applied to right side
- **Transform Pipeline**: DecodeImage → MultiLabelEncode → RecResizeImg → KeepKeys

## 4. Postprocessing Parameters (Section 4.1) - EXACT YAML VALUES

### DB Postprocessing (from det_pp-ocrv5.yml):
```yaml
PostProcess:
  name: DBPostProcess
  thresh: 0.3          # Binary threshold for segmentation
  box_thresh: 0.6      # Confidence threshold for text boxes
  max_candidates: 1000 # Maximum number of contours to process
  unclip_ratio: 1.5    # Box expansion ratio (NOT 2.0 as sometimes documented)
```

### Recognition Postprocessing (from rec_pp-ocrv5.yml):
```yaml
PostProcess:  
  name: CTCLabelDecode  # CTC decoding for text recognition
```

### Loss Functions (from rec_pp-ocrv5.yml):
```yaml
Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:          # For CTC head
    - NRTRLoss:         # For NRTR head
```

## 5. Dependencies (Section 5.1) - COMPLETE LIST WITH YAML REQUIREMENTS

### Core Dependencies (Required):
```python
- cv2 (OpenCV)           # Image processing
- numpy                  # Numerical computations  
- onnxruntime           # ONNX model inference
- shapely               # Geometric operations for DB postprocessing
- pyclipper             # Polygon clipping for text box expansion
```

### Optional Dependencies:
```python
- matplotlib            # For visualization and debugging
- Pillow (PIL)          # Additional image format support
```

### Training Dependencies (from YAML configs):
```python
- paddle                # For model training (not needed for inference)
- visualdl              # For training visualization (use_visualdl: false)
```

### Hardware Requirements (from YAML):
- **GPU Support**: Available (`use_gpu: true` in both configs)
- **CPU Fallback**: Supported for inference-only deployment
- **Distributed Training**: Supported (`distributed: true` in both configs)

## 6. Performance Characteristics (Section 6.1) - BASED ON YAML CONFIGS

### Model Architecture Performance:

#### Detection Model (PP-OCRv5_mobile_det):
- **Backbone**: PPLCNetV3 scale=0.75 (mobile-optimized)
- **Training**: 500 epochs with Cosine learning rate schedule
- **Optimizer**: Adam (β1=0.9, β2=0.999, lr=0.001)
- **Loss**: DBLoss with DiceLoss (α=5, β=10, ohem_ratio=3)
- **Input Constraints**: Images resized to fit in 640x640 (maintaining aspect ratio)

#### Recognition Model (PP-OCRv5_mobile_rec):
- **Backbone**: PPLCNetV3 scale=0.95 (slightly larger for accuracy)
- **Training**: 75 epochs with Cosine learning rate schedule  
- **Optimizer**: Adam (β1=0.9, β2=0.999, lr=0.0005)
- **Text Length**: Maximum 25 characters per text region
- **Input Constraints**: Text regions normalized to height=48px, variable width (min 320px)

### Memory and Processing:
- **Memory**: ~500MB for both models combined
- **Batch Processing**: 
  - Detection: batch_size_per_card=1 (for evaluation)
  - Recognition: batch_size_per_card=128 (for evaluation)

### Character Set and Languages:
- **Dictionary Path**: `./ppocr/utils/dict/ppocrv5_dict.txt` 
- **Character Set**: PP-OCRv5 standard charset with space character support
- **Languages**: Multi-language support (primarily optimized for Chinese, English, Japanese)

## 7. Configuration Files (Section 8.1) - DETAILED YAML ANALYSIS

### Actual Config Files and Their Purposes:

#### Detection Config (`det/det_pp-ocrv5.yml`):
```yaml
Key Settings:
- model_name: PP-OCRv5_mobile_det
- algorithm: DB (Differentiable Binarization)
- d2s_train_image_shape: [3, 640, 640]
- Architecture:
  - Backbone: PPLCNetV3 (scale=0.75, det=True)
  - Neck: RSEFPN (out_channels=96, shortcut=True)  
  - Head: DBHead (k=50, fix_nan=True)
- PostProcess: DBPostProcess (thresh=0.3, box_thresh=0.6, unclip_ratio=1.5)
```

#### Recognition Config (`rec/rec_pp-ocrv5.yml`):
```yaml
Key Settings:
- model_name: PP-OCRv5_mobile_rec
- algorithm: SVTR_LCNet
- d2s_train_image_shape: [3, 48, 320]
- max_text_length: 25
- character_dict_path: ./ppocr/utils/dict/ppocrv5_dict.txt
- use_space_char: true
- Architecture:
  - Backbone: PPLCNetV3 (scale=0.95)
  - Head: MultiHead (CTCHead + NRTRHead)
- PostProcess: CTCLabelDecode
```

#### Additional Files:
- `utils/char_dic.txt` - Character dictionary for recognition
- Model files: `models/det_model.onnx`, `models/rec_model.onnx`

## 8. API Design (Section 9.1) - Simplify

### Actual Usage:
```python
# Complete pipeline
from main import main
result = main()  # Returns formatted OCR results

# Individual components  
from det.rewrite_myself import main_det_run
from rec.rec_inference_onnx import RecognitionONNX

# Detection only
img, boxes = main_det_run()

# Recognition only  
recognizer = RecognitionONNX("models/rec_model.onnx")
text = recognizer.recognize(crop_image)
```

## 9. Architecture Improvements Needed - CRITICAL CORRECTIONS

### Missing Classification Stage (MAJOR ERROR IN DOCUMENTATION):
The original documentation incorrectly describes a 3-stage pipeline:
```
❌ WRONG: Detection → Classification → Recognition (3-stage)
✅ ACTUAL: Detection → Recognition (2-stage)
```

**Evidence from YAML configs:**
- Only 2 model configs exist: `det_pp-ocrv5.yml` and `rec_pp-ocrv5.yml`
- No classification model configuration anywhere
- No cls_model.onnx file in the models directory

### Actual Text Orientation Handling:
Text orientation is handled in the cropping stage via `get_rotate_crop_image()` which auto-rotates if height > 1.5 * width, NOT through a separate classification model.

### Model Architecture Details (from YAML):

#### Detection (DB Algorithm):
- **Purpose**: Text region detection and localization
- **Architecture**: PPLCNetV3 + RSEFPN + DBHead
- **Output**: Probability maps for text regions

#### Recognition (SVTR_LCNet Algorithm): 
- **Purpose**: Character sequence recognition from cropped text regions
- **Architecture**: PPLCNetV3 + MultiHead (CTC + NRTR)
- **Output**: Character sequences with confidence scores

## 10. Model Files Required

### Actual Model Files:
- `models/det_model.onnx` - Detection model
- `models/rec_model.onnx` - Recognition model
- No classification model required

## Recommendations - COMPREHENSIVE DOCUMENTATION REWRITE NEEDED:

### 1. **CRITICAL: Remove all references to classification model/stage**
   - Update architecture diagrams to show 2-stage pipeline only
   - Remove any mention of text orientation classification model
   - Correct all flow diagrams and API documentation

### 2. **Update model specifications with YAML-verified parameters**
   - Detection: [1, 3, 640, 640] with PPLCNetV3 backbone (scale=0.75)
   - Recognition: [1, 3, 48, 320] with PPLCNetV3 backbone (scale=0.95)
   - Include exact architecture details from YAML configs

### 3. **Correct preprocessing normalization ranges and methods**
   - Detection: ImageNet normalization (scale=1./255, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   - Recognition: [-1, 1] range normalization ((pixel/255 - 0.5) / 0.5)

### 4. **Add complete dependency list with geometric processing libraries**
   - Include shapely and pyclipper (critical for DB postprocessing)
   - Specify optional vs required dependencies

### 5. **Update pipeline flow diagram to reflect actual implementation**
   ```
   Input Image → Detection Preprocessing → Detection ONNX → DB Postprocessing → 
   Crop Text Regions → Recognition Preprocessing → Recognition ONNX → CTC Decoding → Final Text
   ```

### 6. **Include exact configuration parameters from YAML files**
   - DB postprocessing: thresh=0.3, box_thresh=0.6, unclip_ratio=1.5
   - Recognition: max_text_length=25, use_space_char=true
   - Architecture details for both models

### 7. **Add validation section with actual model file requirements**
   - Only 2 ONNX files needed: det_model.onnx + rec_model.onnx
   - Character dictionary: ppocrv5_dict.txt
   - No classification model required

### 8. **Performance benchmarks based on YAML training configurations**
   - Include mobile optimization details (PPLCNetV3 scale factors)
   - Batch size recommendations from evaluation configs
   - Hardware requirements (GPU/CPU support)
