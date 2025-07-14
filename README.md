# PaddleOCRv5 ONNX Inference - R\&D Summary Document

## 1. Mục tiêu nghiên cứu

Xây dựng lại pipeline PaddleOCRv5 inference hoàn toàn bằng ONNX, không phụ thuộc vào Paddle framework, để phục vụ nhận dạng chữ từ ảnh (image to text). Nghiên cứu mô hình detection-recognition của PaddleOCRv5, hiểu rõ kiến trúc, chuẩn hóa input/output, cấu hình YAML, và loại bỏ thành phần classifier trong quá trình tối ưu hóa cho inference.

## 2. Tổng quan pipeline

Pipeline inference chia thành hai giai đoạn:

```
Input Image
  → Detection Preprocessing
  → Detection ONNX (DB Algorithm)
  → DB Postprocessing
  → Crop Text Regions
  → Recognition Preprocessing
  → Recognition ONNX (SVTR_LCNet)
  → CTC Decoding
  → Final Text
```

> Ghi chú: Không có bước classification – text orientation được xử lý trong bước crop bằng logic hình học.

## 2.1 Detection Phase
Mục tiêu của bước này là xác định vùng có chứa chữ trong ảnh đầu vào, dưới dạng box 4 điểm.

## 2.1.1 Detection Preprocessing

Trước khi đưa ảnh vào mô hình `PP-OCRv5_mobile_det.onnx`, ảnh cần được biến đổi về định dạng, tỷ lệ và kiểu dữ liệu để đảm bảo khớp hoàn toàn với pipeline huấn luyện gốc. Việc này đảm bảo mô hình hoạt động chính xác, tránh lỗi shape hoặc sai lệch khi suy luận (inference).

Quy trình tiền xử lý bao gồm 5 bước chính như sau:

[Input Image]
   ↓
[Resize to 640x640]
   ↓
[Convert to float32]
   ↓
[Normalize (mean/std)]
   ↓
[HWC → CHW]
   ↓
[Add Batch → [1, 3, 640, 640]]
   ↓
→ Feed into ONNX model

---

### 1. Resize ảnh về kích thước cố định [640, 640]
**Mục đích:**  
Chuyển ảnh về kích thước cố định 640x640 pixel, bất kể kích thước ban đầu.

**Giải thích chi tiết:**

- Mô hình `PP-OCRv5_mobile_det` sử dụng:
  - Backbone: `PPLCNetV3`
  - Detection Head: `DB (Differentiable Binarization)`
  - Ref: https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml

a. Kiến trúc mô hình yêu cầu input cố định `[3, 640, 640]` vì:

  - Các layer như Conv2D, DepthwiseConv, BatchNorm có weight được training theo kích thước này, và ONNX export đã cố định input shape.
  - Khi export sang ONNX (hoặc static inference engine), toàn bộ kernel shape, stride, padding, input/output tensor shape được hard-code.
  - Nếu input sai kích thước, ONNX Runtime sẽ:
    - Báo lỗi shape mismatch, hoặc
    - Chạy sai và tạo ra feature map lệch → hậu xử lý box không chính xác.

b. DB Head phụ thuộc vào tỷ lệ không gian giữa ảnh và output map

  - DB head không trực tiếp predict bounding box, mà sinh ra các map nhị phân:
      - Binary map (text vs background)
      - Threshold map
      - Approximate binarized map

  - Các map này có shape cố định, ví dụ [160 × 160] (do backbone stride = 4)
  - Nếu ảnh input không đúng [640 × 640] thì:
    - Mỗi pixel trên map không còn tương ứng chính xác với vùng ảnh gốc
    - → Decode box bị sai vị trí và scale

Do đó, resize đúng shape là bắt buộc để đảm bảo DB map phản ánh chính xác không gian ảnh gốc.

c. Khác với Recognition, ở bước Detection không cần giữ nguyên aspect ratio khi resize ảnh

  - Việc resize trực tiếp thay vì padding giữ tỉ lệ là một lựa chọn thiết kế
  trong PaddleOCR vì:
    - Detection hoạt động ở cấp độ toàn ảnh (global layout), chứ không cần độ chính xác pixel-level như recognition. Khi resize méo, các đoạn văn bản vẫn giữ được tương quan không gian đủ để model nhận biết vùng có chữ.

  - Kiến trúc DB head không phụ thuộc tuyệt đối vào aspect ratio. Nó học dựa trên hình dạng vùng liên kết (connected region) hơn là chi tiết kích thước chính xác của từng ký tự.
  - Padding giữ tỉ lệ tuy giúp tránh méo hình, nhưng làm chậm inference:
      - Gây thêm thao tác padding/tracking padding size.
      - Cần xử lý ngược padding sau khi decode box.
      - Phức tạp hơn nếu chạy batch-size >1 với nhiều tỉ lệ ảnh khác nhau.

PaddleOCR chấp nhận trade-off: một mức méo nhẹ vẫn đảm bảo detect đủ tốt với đa số văn bản thật, trong khi giúp tăng tốc đáng kể cho inference.

**Ví dụ code**
```python
img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
```

### 2. Convert về `float32`

**Mục đích:**  
Chuyển kiểu dữ liệu từ `uint8` sang `float32` để tương thích với mô hình và các bước xử lý tiếp theo.

**Giải thích chi tiết:**

- ONNX Runtime chỉ hỗ trợ tensor input ở dạng `float32` cho các phép toán như `Conv`, `Mul`, `BatchNorm`.

- Nếu input là `uint8`:
  - ONNX sẽ báo lỗi hoặc thực hiện ép kiểu ngầm → dễ gây lỗi ngầm hoặc kết quả không ổn định.

- Việc normalize sau đó (`/255.0`, trừ `mean`, chia `std`) yêu cầu input là `float32`. Nếu vẫn để `uint8`:
  - Phép chia sẽ trả về `float64` → mismatch kiểu dữ liệu.
  - Hoặc xảy ra phép chia nguyên không chính xác.

**Ví dụ code**
```python
img = img.astype(np.float32)
```

### 3. Chuẩn hóa bằng ImageNet `mean/std`

```python
mean = [0.485, 0.456, 0.406]  
std  = [0.229, 0.224, 0.225]
```
Ref for mean and std: https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml

**Mục đích:**  
Đưa pixel về phân phối chuẩn gần `mean ≈ 0`, `std ≈ 1` như lúc mô hình được huấn luyện trên ImageNet.

2 bước chuẩn hóa ảnh đầu vào:

  - Scale pixel từ [0, 255] → [0.0, 1.0]
  - Normalize ảnh bằng cách trừ mean và chia std của ImageNet, nhằm đưa pixel đầu vào về phân phối có mean ≈ 0 và std ≈ 1 trên từng channel, đúng như mô hình đã được pretrain.

**Giải thích chi tiết:**

a. Backbone `PPLCNetV3` được pretrain trên ImageNet
  - Các trọng số layer (conv, bn, relu) trong PPLCNetV3 được huấn luyện với input có mean/std như trên.
  - Nếu không normalize đúng:
    - Pixel input có phân phối khác → feature map lệch
    - Các bộ lọc (`filters`) học từ ImageNet không còn tương thích
→ Giống như đưa ảnh “nhiễu sáng” hoặc “ngược màu” vào model → mô hình phản ứng sai hoặc cho kết quả rác.

b. Normalize giúp loại bỏ nhiễu ánh sáng và độ tương phản

Ảnh gốc có thể bị tối/sáng, nhiễu, độ tương phản cao thấp không ổn định
Việc normalize giúp:
  - Mỗi pixel mang thông tin tương đối, không tuyệt đối
  - Mô hình tập trung vào biên, cạnh, hình khối (shape) — thứ mà DB head cần để - phân biệt vùng có chữ hay không

c. Tránh sai lệch số học và tăng ổn định khi inference

  - Giá trị pixel nhỏ (≈ ±1) sau normalize giúp tránh:
  - Overflow trong tính toán float
  - Gradient explode/vanish (nếu dùng backward debug)
  - Sai lệch hậu xử lý box nếu scale ảnh bị lệch

**Ví dụ code**
```python
img /= 255.0
img -= np.array(mean, dtype=np.float32)
img /= np.array(std, dtype=np.float32)
```

### 4. Chuyển ảnh từ HWC → CHW

**Mục đích:**  
Đổi thứ tự chiều dữ liệu từ [H, W, C] sang [C, H, W] theo chuẩn đầu vào của ONNX và các framework học sâu.

**Giải thích chi tiết:**

a. Hầu hết các mô hình (Paddle, PyTorch, ONNX) expect input ở dạng `[N, C, H, W]`(với N: batch size, C: số channel, H, W: chiều cao & chiều rộng). Bước này chuẩn bị cho bước tiếp theo — thêm batch dimension — bằng cách đưa channel C lên trước.
- Nếu không chuyển về [C, H, W], việc thêm batch sẽ tạo ra tensor [1, H, W, C], dẫn đến:
  - Shape không hợp lệ cho Conv2D đầu tiên (ONNX sẽ báo lỗi)
  - Swap màu, lỗi output ngầm (silent bug) trong các backend như TensorRT

b. Vì sao Conv2D cần channel C đứng đầu?
  - Các lớp convolution (Conv2D) hoạt động theo cấu trúc:
    - For each channel c:
      Output += Input[c] * Kernel[c]
  - Việc đưa channel lên đầu giúp framework:
      - Truy cập kênh hiệu quả hơn trong memory (data locality tốt hơn)
      - Dễ dàng chia tách per-channel filter khi optimize
      - Hỗ trợ batch operation qua chiều N (batch) phía trước

c. Ngoài ra, một số backend inference không tự báo lỗi rõ
- Với TensorRT, TVM hoặc custom engine: nếu không reshape đúng [C, H, W], bạn có thể bị:
    - Silent failure: ảnh bị swap màu (RGB ↔ BGR)
    - Output rác nhưng không lỗi
    - Debug khó vì không biết do format hay model


**Ví dụ code**
```python
img = img.transpose(2, 0, 1) 
```

### 5. Thêm batch dimension

**Mục đích:**  
Chuyển ảnh từ [C, H, W] sang [1, C, H, W] để mô hình có thể nhận input theo batch.

**Giải thích chi tiết:**

a. ONNX model yêu cầu input có batch dimension:
- Các mô hình ONNX, bao gồm PP-OCRv5_mobile_det, luôn khai báo input với shape `[N, C, H, W]`.
- Nếu thiếu batch dimension:
  - ONNX Runtime sẽ báo lỗi "Invalid input shape".
  - Hoặc sẽ reshape ngầm → dẫn đến lỗi không rõ ràng.

b. Chuẩn bị cho batch inference:
- Việc chuẩn hóa theo batch cũng giúp pipeline dễ mở rộng sau này (batch inference, xử lý nhiều ảnh cùng lúc).

Input shape chính xác yêu cầu:

[1, 3, 640, 640]
    1 → batch size
    3 → RGB
    640 × 640 → spatial dimension

Nếu sai bất kỳ chiều nào:
  - Thiếu batch	-> NNX Runtime báo lỗi Invalid shape
  - Channel ≠ 3	-> Conv layer không khớp weight → lỗi hoặc output rác
  - Size ≠ 640x640 -> Output feature map sai → DB map sai → box sai

**Ví dụ code**
```python
img = np.expand_dims(img, axis=0) 
```

**Ghi chú thêm:**  
Trong PaddleOCR, batch dimension được thêm tự động ở tầng `loader:`.  
Tuy nhiên, khi viết pipeline inference ONNX riêng, bạn **phải thêm thủ công** batch `[1, C, H, W]`trước khi đưa vào `session.run()`, nếu không sẽ gặp lỗi shape.


## 2.1.2 Detection Inference (PP-OCRv5 det.onnx – DB Algorithm)
Sau khi ảnh đầu vào đã được tiền xử lý thành tensor [1, 3, 640, 640], bước tiếp theo là chạy mô hình ``PP-OCRv5_mobile_det.onnx`` bằng ONNX Runtime để sinh ra DB probability map — bản đồ xác suất vùng chứa văn bản làm, nền tảng cho bước Postprocessing → decode polygon box.

Tổng quan pipeline:

[Preprocessed Image: [1, 3, 640, 640]]
        ↓
[Run ONNX Session]
        ↓
[Output: [1, 1, 160, 160] (DB Probability Map)]
        ↓
→ Gửi sang bước Postprocessing

**Mục tiêu** 
Sinh ra một bản đồ xác suất cho toàn ảnh. Mỗi pixel trong map tương ứng với xác suất "vùng đó chứa text".

**Tư tưởng kiến trúc DB Text Detection**
- Khác với các phương pháp detection truyền thống (SSD, YOLO, FasterRCNN) vốn học cách regress toạ độ bounding box trực tiếp, DB (Differentiable Binarization) học một bản đồ phân đoạn nhị phân (binary segmentation map) cho chữ.
- Lý do chọn segmentation:
  - Text có thể rất mảnh, nối liền, độ dài khác nhau, hoặc không có hình chữ nhật cố định.
  - Box regression dễ sai khi gặp text xiên, dài hoặc quá gần nhau.
  - Phân đoạn ra vùng chữ cho phép xử lý linh hoạt hơn với các hậu xử lý như polygon fitting.

## 1. Input & Output Shape
- Input: [1, 3, 640, 640] -> Ảnh RGB đã resize, normalize, CHW format, thêm batch dimension
  - 1: batch size
  - 3: RGB
  - 640x640: ảnh đã resize
- Output: [1, 1, 160, 160] -> DB map (1 channel), mỗi pixel là xác suất của một vùng chứa văn bản trong ảnh
  - 1: batch size
  - 1: single channel — probability map
  - 160x160: spatial map (do stride tổng = 4)

## 2. Cấu trúc nội tại của mô hình PP-OCRv5_mobile_det.onnx
Mô hình chia thành 3 phần chính:
  - Backbone: PPLCNetV3 -> Trích xuất đặc trưng từ ảnh
  - Neck: RSEFPN -> Tổng hợp multi-scale features
  - Head: DBHead -> Dự đoán bản đồ xác suất vùng chứa chữ

## 2.1 PPLCNetV3
  - Là backbone nhẹ, thiết kế dành cho mobile.
  - Sử dụng nhiều Depthwise Separable Convs, kết hợp SE modules.
  - Feature maps được trích xuất ở các tầng stride = {1, 2, 4, 8...}.

PaddleOCR lấy output tại stride=4: nghĩa là ảnh input 640x640 sẽ sinh ra feature 160x160 (giảm 4 lần).

**Tại sao lại lấy output tại stride=4?**
1. Text trong ảnh có kích thước rất nhỏ
  - Trong ảnh tự nhiên (billboard, hóa đơn, biển hiệu...), các ký tự có thể chỉ chiếm vài pixel.
  - Nếu trích xuất feature từ tầng stride=8 hoặc stride=16:
    - Pixel trong feature map đại diện cho vùng 8×8 hoặc 16×16 trong ảnh gốc
    - Rất dễ mất chi tiết chữ nhỏ, đặc biệt là các nét mảnh, những nét dấu nhỏ...

2. DB algorithm cần độ phân giải cao để tạo biên rõ ràng
  - DB không học box, mà học biên chữ (boundary-aware).
  - Càng có nhiều pixel ở gần biên, segmentation map càng chính xác.
  - Nếu dùng stride cao → hình dạng chữ sẽ bị co méo → polygon fitting sẽ sai.
→ Độ phân giải cao ở output (160x160) giúp hậu xử lý (DBPostProcess) detect được hình dạng chữ sát thực tế hơn.

3. Tối ưu giữa độ chính xác và chi phí tính toán
  - Nếu lấy feature tại stride=1 hoặc 2 → output sẽ là 640×640 hoặc 320×320 → cực kỳ nặng.
  - Stride=4 là điểm cân bằng tốt:
    - Vẫn giữ được chi tiết
    - Vẫn đủ nhẹ cho inference real-time, đặc biệt trên mobile
  → Đó là lý do PP-OCRv5_mobile_det dùng PPLCNetV3 + feature ở stride=4

## 2.2 RSEFPN — Residual Squeeze-and-Excitation Feature Pyramid Network

**Vai trò**
Là "neck" của mô hình, RSEFPN có nhiệm vụ:
  - Kết hợp các feature map từ nhiều tầng (multi-scale)
  - Tăng khả năng nhận diện chữ ở nhiều kích cỡ
  - Không làm thay đổi độ phân giải không gian (giữ nguyên [160, 160])

**Tại sao phải dùng FPN?**
  - Text trong ảnh có thể rất nhỏ hoặc lớn tùy ngữ cảnh.
  - Backbone tạo ra feature ở nhiều cấp độ, mỗi cấp mạnh ở 1 loại chữ:
    - Feature sâu → mạnh với object lớn
    - Feature nông → tốt cho chi tiết nhỏ
  → Nếu chỉ dùng 1 cấp → sẽ fail 1 nhóm chữ nào đó.

**FPN giải quyết thế nào?**
  - Top-down + lateral connections: lấy feature từ nhiều tầng → resize → align → sum lại
  - Làm cho mô hình "nhìn được cùng lúc" các scale khác nhau của text

**Tại sao gọi là RSE*FPN?**
  - PaddleOCR thêm Squeeze-and-Excitation (SE) blocks để học được importance của từng channel
  - "Residual" = thêm shortcut connection → giúp gradient ổn định hơn

**Output**
  - Output feature giữ nguyên kích thước: [1, C, 160, 160]
  - Channel C được tổng hợp từ nhiều tầng, nhưng spatial vẫn là stride=4
  → Sẵn sàng đưa vào DBHead để sinh map xác suất

## 2.3 DBHead — Phân đoạn chữ bằng mô hình nhẹ

**Vai trò**
- Dự đoán một probability map, mỗi pixel ∈ [0, 1], là xác suất pixel đó thuộc về vùng chữ.
- Không giống các object detector thông thường (YOLO, SSD...) học toạ độ box, DBHead học rìa chữ bằng segmentation.

**Kiến trúc đơn giản**
  Conv 3×3 → BatchNorm → ReLU → Conv 1×1 → Sigmoid
Cụ thể:
  - Conv 3×3	Học local features
  - BN + ReLU	Normalize, nonlinear
  - Conv 1×1	Giảm về 1 channel
  - Sigmoid	Đưa output ∈ [0,1] — xác suất chữ

**Tại sao dùng segmentation thay vì box?**
- Chữ thường dính sát, xoay nghiêng, dài không đều, rất khó dùng box để bao
- DBHead được thiết kế để:
  - Dự đoán pixel-level, chứ không object-level
  - Học được vùng biên mềm → càng gần biên càng chắc chắn
  - Sau đó hậu xử lý bằng threshold → mask → polygon

**Boundary-Aware Training (DB)**
- DB paper dùng Differentiable Binarization (DB):
  - Dự đoán 2 map: prob_map, threshold_map
  - Áp dụng sigmoid(b(x)) → "soft binarization"
  - PaddleOCR V5 tối giản chỉ dùng 1 map (prob_map), hậu xử lý cứng (DBPostProcess)
→ Vẫn giữ được hiệu quả nhưng inference nhanh hơn

Ref: https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml
     https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/det_mobilenet_v3.py

## 3 ONNXRuntime Inference
Dưới đây là đoạn code mô phỏng cách thực thi mô hình:
```python
import onnxruntime as ort

session = ort.InferenceSession("PP-OCRv5_mobile_det.onnx")
input_name = session.get_inputs()[0].name

output = session.run(None, {input_name: input_tensor})
```
Thông tin tensor:
```python
  - input_tensor: np.ndarray — shape [1, 3, 640, 640], dtype float32
  - output[0]: np.ndarray — shape [1, 1, 160, 160], dtype float32
```

## 4. Vì sao output là [1, 1, 160, 160]?
  - Mô hình chỉ downsample input duy nhất 4 lần (stride=4) → tránh mất thông tin hình dạng chữ.
  - RSEFPN và DBHead giữ nguyên kích thước → không có upsample/downsample thêm.
  - Output cuối là 1 channel từ DBHead: Conv → Sigmoid
  → Mỗi pixel trong map là 1 điểm trên ảnh feature 160x160, tương ứng với vùng 4×4 trong ảnh gốc 640x640.

## 5. Ý nghĩa của DB Map
  - Output: [1, 1, 160, 160]
    - Là một heatmap xác suất nhị phân
    - Giá trị gần 1: vùng nhiều khả năng chứa text
    - Giá trị gần 0: background
  - Chưa thể dùng trực tiếp — cần qua hậu xử lý.
  
## 3. Thành phần chi tiết

### 3.1 Detection Model (PP-OCRv5\_mobile\_det)

* **Kiến trúc chính**: DB (Differentiable Binarization)
* **Backbone**: PPLCNetV3, scale=0.75
* **Neck**: RSEFPN, 96 kênh, shortcut=True
* **Head**: DBHead, k=50, fix\_nan=True
* **Input**: \[1, 3, 640, 640]
* **Output**: Probability map \[1, 1, H, W]

#### Vì sao chọn DB:

* Phù hợp với bài toán segment vùng text (thay vì detect box cứng)
* Kết quả ra dạng mask → dễ postprocess thành box chính xác

### 3.2 Recognition Model (PP-OCRv5\_mobile\_rec)

* **Kiến trúc chính**: SVTR\_LCNet
* **Backbone**: PPLCNetV3, scale=0.95
* **Head**: MultiHead (CTCHead + NRTRHead)
* **SVTR Neck**: dims=120, depth=2, hidden\_dims=120
* **Input**: \[1, 3, 48, variable-width]
* **Output**: Sequence \[1, T, vocab\_size]
* **Giới hạn độ dài**: max\_text\_length = 25

#### Vì sao dùng SVTR\_LCNet:

* Kết hợp CNN (LCNet) với self-attention (SVTR) → nhẹ, chính xác
* Phù hợp thiết bị mobile, inference nhanh

### 3.3 Text Orientation Handling

* Không dùng classification model
* Góc xoay được xử lý trong hàm `get_rotate_crop_image()`
* Logic: Nếu box có height > 1.5 \* width → tự động xoay dọc

## 4. Xử lý ảnh và cấu hình YAML

### 4.1 Detection Preprocessing

* Resize về \[3, 640, 640], scale theo tỉ lệ ảnh gốc
* Normalize theo ImageNet:

  ```yaml
  
  scale: 1./255.
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
  ```

### 4.2 Recognition Preprocessing

* Resize chiều cao = 48px, width biến đổi theo tỉ lệ ảnh (min = 320)
* Normalize: (pixel / 255 - 0.5) / 0.5 → range \[-1, 1]
* Padding bên phải nếu width chưa đủ

### 4.3 Postprocessing Detection (DBPostProcess)

```yaml
thresh: 0.3
box_thresh: 0.6
max_candidates: 1000
unclip_ratio: 1.5
```

### 4.4 Postprocessing Recognition (CTCLabelDecode)

* Dùng CTC decoding để tạo chuỗi ký tự từ xác suất frame
* Dictionary: `ppocrv5_dict.txt`
* Hỗ trợ tiếng Trung, Nhật, Anh, ký tự đặc biệt

## 5. Cấu hình huấn luyện và khả năng mở rộng

### Detection Training (theo YAML)

* Optimizer: Adam (lr=0.001)
* Epochs: 500, Cosine LR
* Loss: DBLoss (α=5, β=10)

### Recognition Training

* Optimizer: Adam (lr=0.0005)
* Epochs: 75, Cosine LR
* Loss: MultiLoss (CTCLoss + NRTRLoss)

### Batch Size

* Detection: 1 (eval)
* Recognition: 128 (eval)

## 6. Phân tích điểm mạnh / hạn chế

### Điểm mạnh:

* Lightweight, tốc độ nhanh, chính xác tốt
* Không phụ thuộc Paddle khi convert sang ONNX
* Có thể chạy hoàn toàn bằng `onnxruntime` + `numpy`

### Hạn chế:

* Không có stage classification → chưa xử lý tốt text nghiêng ngược
* SVTR mặc định dùng dict gốc Trung Quốc – cần thay dict nếu muốn dùng tiếng Việt
* Width recognition phải >=320px → ảnh nhỏ dễ bị pad trắng

## 7. Kiến trúc thư mục gợi ý

```
project_root/
├── models/
│   ├── det_model.onnx
│   └── rec_model.onnx
├── dict/ppocrv5_dict.txt
├── pipeline/
│   ├── preprocess.py
│   ├── detect.py
│   ├── crop.py
│   ├── recognize.py
│   └── postprocess.py
├── main.py
└── README.md
```

## 8. Phụ lục

### 8.1 Thư viện phụ thuộc

```bash
opencv-python
numpy
onnxruntime
shapely
pyclipper
matplotlib (optional)
Pillow (optional)
```

### 8.2 File cấu hình YAML chính

* det/det\_pp-ocrv5.yml
* rec/rec\_pp-ocrv5.yml

### 8.3 Mô tả dictionary ký tự

* `ppocrv5_dict.txt`: Gồm >7000 ký tự: chữ Trung, Nhật, Latin, số, ký hiệu, khoảng trắng

---

**Người thực hiện:** \[Tên bạn]
**Ngày hoàn tất:** \[DD/MM/YYYY]
**Mục đích:** Lưu trữ tri thức nội bộ, phục vụ future dev/debug/integration
