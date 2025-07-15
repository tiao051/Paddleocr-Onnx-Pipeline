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

### 2.1 Detection Phase
Mục tiêu của bước này là xác định vùng có chứa chữ trong ảnh đầu vào, dưới dạng box 4 điểm.

#### 2.1.1 Detection Preprocessing
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

##### 1. Resize ảnh về kích thước cố định [640, 640]
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

  - Backbone có `stride = 4` → output feature map size = `640 / 4 = 160`
  - Tức là mỗi pixel trên DB map [160×160] tương ứng một vùng 4×4 pixel trên ảnh gốc.
  - Nếu input không phải 640x640, mapping này sẽ sai → box cuối cùng lệch vị trí và scale.

Do đó, resize đúng shape là bắt buộc để đảm bảo DB map phản ánh chính xác không gian ảnh gốc.

c. Khác với Recognition, ở bước Detection không cần giữ nguyên aspect ratio khi resize ảnh

  - Việc resize trực tiếp thay vì padding giữ tỉ lệ là một lựa chọn thiết kế
  trong PaddleOCR vì:
    - Detection hoạt động ở cấp độ toàn ảnh (global layout), chứ không cần độ chính xác pixel-level như recognition. Khi resize méo, các đoạn văn bản vẫn giữ được tương quan không gian đủ để model nhận biết vùng có chữ.

  - Kiến trúc DB head không phụ thuộc tuyệt đối vào aspect ratio. Nó học dựa trên hình dạng vùng liên kết (connected region) hơn là chi tiết kích thước chính xác của từng ký tự.
  - Padding giữ tỉ lệ tuy giúp tránh méo hình, nhưng làm chậm inference:
      - Gây thêm thao tác padding/tracking padding size.
      - Cần xử lý ngược padding sau khi decode box.
      - Phức tạp hơn nếu chạy batch-size > 1 với nhiều tỉ lệ ảnh khác nhau.

⟶ PaddleOCR chọn resize trực tiếp (không giữ tỷ lệ) để đơn giản hóa pipeline và tăng tốc inference, chấp nhận một mức méo nhẹ nhưng không ảnh hưởng tới chất lượng detect.  

**Ví dụ code**
```python
img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
```

##### 2. Convert về `float32`

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

##### 3. Chuẩn hóa bằng ImageNet `mean/std`

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

##### 4. Chuyển ảnh từ HWC → CHW

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

##### 5. Thêm batch dimension

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

#### 2.1.2 Detection Inference (PP-OCRv5 det.onnx – DB Algorithm)
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

##### 1. Input & Output Shape
- Input: [1, 3, 640, 640] -> Ảnh RGB đã resize, normalize, CHW format, thêm batch dimension
  - 1: batch size
  - 3: RGB
  - 640x640: ảnh đã resize
- Output: [1, 1, 160, 160] -> DB map (1 channel), mỗi pixel là xác suất của một vùng chứa văn bản trong ảnh
  - 1: batch size
  - 1: single channel — probability map
  - 160x160: spatial map (do stride tổng = 4)

##### 2. Cấu trúc nội tại của mô hình PP-OCRv5_mobile_det.onnx
Mô hình chia thành 3 phần chính:
  - Backbone: PPLCNetV3 -> Trích xuất đặc trưng từ ảnh
  - Neck: RSEFPN -> Tổng hợp multi-scale features
  - Head: DBHead -> Dự đoán bản đồ xác suất vùng chứa chữ

###### 2.1 PPLCNetV3
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

###### 2.2 RSEFPN — Residual Squeeze-and-Excitation Feature Pyramid Network

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

###### 2.3 DBHead — Phân đoạn chữ bằng mô hình nhẹ

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

##### 3 ONNXRuntime Inference
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

###### 3.1 Vì sao output là [1, 1, 160, 160]?
  - Mô hình chỉ downsample input duy nhất 4 lần (stride=4) → tránh mất thông tin hình dạng chữ.
  - RSEFPN và DBHead giữ nguyên kích thước → không có upsample/downsample thêm.
  - Output cuối là 1 channel từ DBHead: Conv → Sigmoid
  → Mỗi pixel trong map là 1 điểm trên ảnh feature 160x160, tương ứng với vùng 4×4 trong ảnh gốc 640x640.

###### 3.2 Ý nghĩa của DB Map
  - Output: [1, 1, 160, 160]
    - Là một heatmap xác suất nhị phân
    - Giá trị gần 1: kvùng nhiều khả năng chứa text
    - Giá trị gần 0: background
  - Chưa thể dùng trực tiếp — cần qua hậu xử lý.

#### 2.1.3 Detection Postprocessing (DBPostProcess)

Sau khi mô hình ONNX đã sinh ra DB probability map [1, 1, 160, 160], bước tiếp theo là chuyển đổi bản đồ xác suất này thành các bounding box 4 điểm (quadrilateral) chứa văn bản, sẵn sàng cho bước crop và recognition.

Tổng quan pipeline:

[DB Probability Map: [1, 1, 160, 160]]
        ↓
[Binary Thresholding → Binary Mask]
        ↓
[Contour Detection → List of Contours]
        ↓
[Box Extraction & Filtering → Raw Boxes]
        ↓
[Unclip Expansion → Expanded Boxes]
        ↓
[Scale to Original Image → Final Boxes]
        ↓
→ Output: boxes [[x1,y1,x2,y2,x3,y3,x4,y4], ...] + scores [0.85, 0.79, ...]

**Mục tiêu**
Chuyển đổi soft probability map thành hard bounding boxes với confidence scores, giữ nguyên độ chính xác vị trí văn bản trong ảnh gốc.

**Cấu hình**
PostProcess:
  name: DBPostProcess
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5

##### 1. Binary Thresholding

**Mục đích:**  
DB head sinh ra bản đồ xác suất với mỗi pixel ∈ [0,1], thể hiện xác suất điểm đó thuộc vùng text. Tuy nhiên, mask liên tục không thể dùng cho các thuật toán xử lý hình học.

Do đó, bước đầu tiên là nhị phân hóa (thresholding) để tạo ảnh nhị phân ∈ {0,1}, giúp xác định rạch ròi vùng nào là text, vùng nào là background.

**Giải thích chi tiết:**

a. Áp dụng threshold cố định
```python
thresh = 0.3
binary_mask = (prob_map > thresh).astype(np.uint8)
```

- Duyệt qua từng pixel trong map, so sánh với ngưỡng cố định (e.g. 0.3).
- Pixel nào có giá trị lớn hơn ngưỡng → gán giá trị 1 (text); ngược lại → 0 (background).
- Kết quả là một binary mask cùng shape với map gốc, thường chuyển sang kiểu `uint8` để dễ xử lý về sau.

**Vì sao phải threshold?**
- Các thuật toán như contour extraction, connected component analysis chỉ hoạt động khi ranh giới rõ (binary).
- Nếu để mask dạng float, hệ thống không thể xác định biên hoặc vùng liên thông.

**Vì sao threshold thường là 0.3?**
- Thực nghiệm trên nhiều domain (hóa đơn, scan, scene text) cho thấy 0.3 cân bằng tốt giữa:
  - False positive: quá thấp → noise cũng thành text
  - False negative: quá cao → mất nét chữ mảnh

**Tác động đến downstream:**
- Threshold ảnh hưởng trực tiếp đến kích thước box được fitting phía sau:
  - Quá thấp → mask phình → box overshoot
  - Quá cao → mask co lại → box thiếu nét

**(Optional)**: Có thể áp dụng phép dilation để nối những vùng text đứt đoạn, nhưng cần test kỹ với domain thật:
- Scene text thường vỡ → nên dilation nhẹ
- Document scan thường clean → không cần dilation

##### 2. Contour Detection

**Mục đích:**  
Xác định các vùng chứa văn bản tiềm năng từ binary mask, bằng cách gom nhóm các pixel trắng (giá trị = 1) thành các vùng liên thông độc lập.

**Bản chất**  
- Là quá trình phân tích **các vùng liên thông (connected components)** trong ảnh nhị phân.
- Dựa trên tính **liên thông trong lưới pixel** – 4-neighbor hoặc 8-neighbor.
- Kết quả là tập hợp các đường biên (contour) bao quanh mỗi vùng text tiềm năng.

**How it works:**
- Duyệt từng pixel trong binary mask để xác định các nhóm pixel liền kề nhau (connected region).
- Hai phương pháp liên thông phổ biến:
  - **4-neighbor**: pixel kết nối với trái, phải, trên, dưới
  - **8-neighbor**: thêm kết nối chéo → đảm bảo phát hiện vùng tốt hơn nếu chữ bị nghiêng/mảnh

- Mỗi nhóm liên thông được đánh nhãn (region ID) → sinh ra contour tương ứng:
  - **Contour** là tập hợp các điểm nằm ở biên ngoài của vùng mask.
  - Contour có thể biểu diễn đầy đủ (tất cả biên) hoặc tối giản (giữ các điểm góc).

- Output: danh sách contour ứng với các vùng text riêng biệt → làm input cho bước polygon fitting.

**Tại sao cần bước này?**
- Mỗi vùng contour là đại diện hình học của một cụm văn bản (từ, dòng, đoạn).
- Không có contour → không thể fit polygon → không thể crop để nhận dạng.
- Là cầu nối giữa ảnh nhị phân và box thực tế.

**Yêu cầu kỹ thuật:**
- Ảnh input **bắt buộc là binary (dạng uint8)**:
  - Nếu vẫn còn giá trị float (e.g. 0.2, 0.7): không có ngưỡng rõ → nối nhầm vùng
  - Kết quả: sai biên → fit box sai → crop sai chữ → rec lỗi

**Thiết kế thuật toán – abstraction không phụ thuộc OpenCV:**

Kỹ thuật thiết kế: 
- Flat Retrieval (non-hierarchical) -> Text thường là các cụm độc lập → không cần phân tầng cha–con
- Contour Simplification -> Chỉ giữ các điểm góc (corner) thay vì toàn bộ đường biên → giảm độ phức tạp
- Ignore Inner Nesting -> Vùng text lồng nhau rất hiếm trong OCR thực tế → không cần hierarchy

**Ghi chú cho production:**
- Với ảnh scan văn bản hoặc hóa đơn → các vùng thường có biên rõ, ít dính nhau → 4-neighbor đủ.
- Với ảnh scene hoặc chữ tay → nhiều nét chéo, nét mảnh → nên dùng 8-neighbor để tránh vỡ vùng.
- Contour là bước rất nhạy với noise từ bước threshold trước đó:
  - Nếu threshold thấp quá → vùng dính nhau → contour gộp
  - Nếu threshold cao quá → vùng đứt nét → contour không khép kín

**Tóm lại**
Contour Detection là cầu nối từ binary mask sang box hình học. Nếu threshold sai hoặc ảnh quá nhiễu, bước này dễ fail nhất trong pipeline DBPostProcess.

##### 3. Box Extraction & Filtering

**Mục đích:**  
Chuyển mỗi contour thành một bounding box đơn giản (thường là quadrilateral), sau đó lọc ra các box hợp lệ dựa trên score và hình dạng để đảm bảo chất lượng đầu ra cho bước crop và recognition.

Pipeline nội bộ:
[Contour] → [Polygon Fitting] → [Confidence Scoring] → [Box Filtering]

**Quy trình gồm 2 bước chính:**

1. **Polygon fitting (box extraction)** – tìm đa giác bao quanh vùng text
2. **Filtering & scoring** – tính độ tin cậy, loại bỏ box nhiễu

**1. Polygon Fitting (Box extraction)**  

**Lý thuyết & Toán học:**

- **Mục tiêu**
Chuyển mỗi contour (vùng liên thông trên binary mask) thành một đa giác đơn giản (thường là tứ giác/quadrilateral) bao sát vùng text.

**Bản chất toán học**
Cho một tập hợp điểm biên 𝐶={(𝑥𝑖,𝑦𝑖)} của contour, bài toán là tìm một polygon 𝑃(thường là 4 điểm) sao cho:
  - 𝑃 bao trọn 𝐶 (containment)
  - Diện tích 𝑃 nhỏ nhất có thể (tight fit)
  - Hình học đơn giản để dễ crop (thường là rotated rectangle hoặc convex hull)

**Các phương pháp phổ biến**
  1. **Rotated Rectangle (MinAreaRect):**
    - Tìm hình chữ nhật xoay có diện tích nhỏ nhất bao trọn contour.
    - Ưu điểm: Đơn giản, nhanh, luôn ra 4 điểm, phù hợp với text nằm ngang/nghiêng.

  2. **Polygon Approximation (Douglas-Peucker):**
    - Giảm số điểm của contour thành polygon ít đỉnh hơn (thường là 4).
    - Nếu ra đúng 4 điểm → dùng luôn, nếu không → fallback về rotated rect.

  3. **Convex Hull:**
    - Lấy bao lồi của contour, có thể nhiều hơn 4 điểm.
    - Thường chỉ dùng khi contour quá phức tạp.

**Tại sao cần polygon fitting?**
  - Đơn giản hóa vùng mask thành hình học dễ xử lý (crop, transform).
  - Giảm nhiễu, loại bỏ các chi tiết nhỏ không liên quan.
  - Đảm bảo box có thể dùng trực tiếp cho recognition (4 điểm → perspective transform).

**Lọc box nhỏ/noise:**
  - Tính độ dài cạnh ngắn nhất của polygon:
    min_side=min(|𝑃1-𝑃2|, |𝑃2-𝑃3|,..)
​  - Nếu min_side < min_box_size → loại (noise).

**Thứ tự điểm:**  
  - Sắp xếp 4 điểm theo thứ tự (top-left, top-right, bottom-right, bottom-left) để chuẩn hóa cho bước crop.
  
**2. Confidence Scoring**  

**Mục tiêu:**  
Đánh giá độ tin cậy (confidence) của mỗi box bằng cách tính trung bình xác suất (probability) của các pixel nằm trong vùng polygon (box) trên DB probability map.

**Bản chất**
- Gọi `P(x, y)` là xác suất tại pixel `(x, y)` trên DB map, `Ω` là tập hợp các pixel nằm trong polygon box.  
  Khi đó: Score_box = (1 / |Ω|) * ∑ {(x,y) ∈ Ω} P(x, y)
  - `|Ω|`: Số pixel bên trong polygon  
  - `∑ P(x, y)`: Tổng xác suất các pixel trong polygon  

**Hai cách tính thực tế:**
  1. **Fast Mode (gần đúng):**
    - Lấy mean xác suất trong bounding rectangle của box (không cần mask polygon).
    - Nhanh, đủ chính xác cho hầu hết trường hợp thực tế.
      - Công thức:
        ```python
        xmin, xmax = int(box[:, 0].min()), int(box[:, 0].max())
        ymin, ymax = int(box[:, 1].min()), int(box[:, 1].max())
        score = prob_map[ymin:ymax, xmin:xmax].mean()
        ```
  2. **Slow Mode (chính xác):**
    - Tạo mask đúng hình polygon, chỉ lấy mean các pixel thực sự nằm trong polygon.
    - Chính xác hơn, nhưng chậm hơn do phải tạo mask.
      - Công thức:
        ```python
        mask = np.zeros_like(prob_map, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 1)
        score = prob_map[mask == 1].mean()
        ```
**Ý nghĩa xác suất:**  
  - Nếu box nằm trọn vùng text, các 𝑃(x, y) sẽ gần 1 → score cao.
  - Nếu box nằm vùng background/noise, các 𝑃(x, y) sẽ gần 0 → score thấp.
  - Score này là **ước lượng xác suất trung bình** vùng box chứa text.

**Lọc theo area và score:**  
  - Nếu diện tích box quá nhỏ (ví dụ cạnh < 3px) → loại (noise).
  - Nếu score < box_thresh (ví dụ 0.6) → loại (nhiễu).
  - Nếu đạt cả hai điều kiện trên → giữ lại box này cho output.

**Tóm lại:**
Confidence scoring là bước định lượng xác suất một box thật sự chứa văn bản. Dù dùng mode fast hay slow, mục tiêu là đảm bảo chỉ giữ lại các vùng có độ tin cậy cao cho recognition.

**Input/Output Summary**
- Polygon Fitting:
  - Input: List các contour (mỗi contour là tập hợp điểm biên)- Binary mask shape [H, W]
  - Output: List các polygon (đa giác 4 điểm) dạng [[x1,y1,x2,y2,...], ...]
- Confidence Scoring: 
  - Input: Mỗi polygon (4 điểm)- DB probability map shape [1, 1, H, W]
  - Output: Mỗi polygon kèm theo score (confidence)
- Box Filtering:
  - Input: Polygon + score; Tham số: min_box_size, box_thresh
  - Output: Các box hợp lệ: [[x1,y1,x2,...,x4,y4], ...] kèm scores tương ứng

Lưu ý: 
  - Shape của prob_map luôn là [1, 1, H, W], cần squeeze thành [H, W] để dùng.
  - Mỗi box giữ lại đều đã được sắp xếp theo thứ tự chuẩn 4 điểm (top-left → clockwise).

##### 4. Unclip Expansion

**Mục tiêu:**  
Mở rộng polygon box ra ngoài contour ban đầu để đảm bảo bao trọn toàn bộ vùng text, tránh crop thiếu ký tự ở biên, đặc biệt khi DB probability map có xu hướng "co" vùng text lại nhỏ hơn thực tế.

**Lý thuyết & Toán học:** 
- Sau khi fitting polygon (thường là tứ giác), box này có thể chưa bao hết nét chữ thật do đặc tính conservative của DBNet.
- Unclip là phép **offset đều polygon ra ngoài** một khoảng xác định, tạo box lớn hơn nhưng vẫn giữ hình dạng gốc.
- Ý tưởng dựa trên kỹ thuật offset đường biên (polygon offsetting) bằng vector pháp tuyến (normal vector).
- Nếu 𝐴 là diện tích polygon và 𝐿 là chu vi:
    distance = (𝐴.(𝑟2−1))/𝐿 
  với 𝑟 = unclip ratio (thường từ 1.5–2.0)

**Thực thi:**  
- Dùng thư viện `pyclipper` để offset polygon:
  ```python
  import pyclipper
  from shapely.geometry import Polygon

  poly = Polygon(box)
  distance = poly.area * unclip_ratio / poly.length

  offset = pyclipper.PyclipperOffset()
  offset.AddPath(box.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
  expanded = offset.Execute(distance)
  if len(expanded) > 0:
      expanded_box = np.array(expanded[0])
  else:
      expanded_box = box
  ```
- **JT_ROUND** giúp các góc bo tròn, tránh tạo ra các đỉnh sắc nhọn bất thường.

**Tại sao cần unclip?**
  - DB probability map có xu hướng "co" text region để tránh false positive, dẫn đến box nhỏ hơn thực tế.
  - Nếu không unclip, khi crop sẽ dễ bị thiếu nét chữ ở biên, giảm accuracy recognition.
  - Unclip giúp tăng recall mà không làm tăng nhiều false positive nếu chọn unclip_ratio hợp lý.

**Chọn unclip_ratio bao nhiêu là hợp lý?**
Từ nghiên cứu thực nghiệm trên DBNet paper:
  - unclip_ratio < 1.2 → crop thiếu ký tự biên (phù hợp với ảnh scan, text rõ nét)
  - unclip_ratio > 2.0 → crop quá nhiều noise xung quanh (chỉ dùng khi text rất mờ hoặc bị vỡ nét)
  - 1.5 là sweet spot phù hợp với ảnh thực tế, nhiều noise, text mảnh

**Tóm lại:**  
Unclip expansion là bước mở rộng polygon box dựa trên hình học, đảm bảo vùng crop bao trọn text thật, là chìa khóa để tăng độ chính xác nhận dạng trong pipeline OCR thực tế.

##### 5. Scale to Original Image

**Mục đích:**  
Chuyển đổi coordinates từ detection resolution (160×160) hoặc detection input (640x640) về original image resolution.

**Lý thuyết & Toán học:**  
- Sau khi postprocess, các box thường nằm ở scale của feature map (160×160) hoặc detection input (640×640).
- Ảnh gốc có thể có kích thước bất kỳ (H_orig × W_orig).
- Cần scale lại toạ độ box về đúng tỷ lệ ảnh gốc.

**Cách tính:**  
Giả sử:
- Ảnh gốc: (H_orig, W_orig)
- Detection input: (640, 640)
- Feature map: (160, 160) (stride=4)

Các bước:
1. **Scale từ feature map lên detection input:**  
   - Nếu box lấy từ feature map (160×160):  
     ```python
     box[:, 0] *= 4  # x
     box[:, 1] *= 4  # y
     ```
2. **Scale từ detection input về ảnh gốc:**  
   - Tính tỉ lệ scale:
     ```python
     scale_h = H_orig / 640
     scale_w = W_orig / 640
     box[:, 0] *= scale_w
     box[:, 1] *= scale_h
     ```
3. **Clip toạ độ để không vượt ngoài ảnh:**
   ```python
   box[:, 0] = np.clip(box[:, 0], 0, W_orig)
   box[:, 1] = np.clip(box[:, 1], 0, H_orig)
   ```

**Tại sao phải scale lại?**
- Nếu không scale, box sẽ crop sai vị trí trên ảnh gốc (bị lệch hoặc méo).
- Đảm bảo mọi bước hậu xử lý đều trả về kết quả đúng với không gian ảnh ban đầu.

**Tóm lại:**  
Scale to Original Image là bước cuối cùng trong DBPostProcess, đảm bảo các box 4 điểm trả về đúng vị trí thực tế trên ảnh gốc, sẵn sàng cho bước crop và recognition.

**Ghi chú quan trọng:**
- DBPostProcess là bước **quan trọng nhất** quyết định chất lượng detection
- Các tham số `thresh`, `box_thresh`, `unclip_ratio` cần tune theo từng loại ảnh
- Trade-off giữa speed và accuracy: fast mode vs slow mode scoring

### 2.2 Crop Text Regions (Perspective Crop)

**Mục đích:**  
Cắt từng vùng text từ ảnh gốc dựa trên box 4 điểm đã detect, chuẩn hóa orientation để chuẩn bị cho bước recognition.

**Pipeline**
[Original Image] + [List of 4-point Boxes]
        ↓
[Perspective Transform]
        ↓
→ Output: List of Cropped Patches (rectified text regions)

**Lý thuyết & Toán học:**  
Mỗi box là một polygon 4 điểm (quadrilateral), có thể nghiêng, méo hoặc không song song trục ảnh, không thể dùng trực tiếp cho recognition.
  - Recognition model (như CRNN, SVTR) chỉ hoạt động tốt khi chữ nằm ngang, vuông góc.
  - Nếu crop bằng bounding box hoặc clip đơn thuần → chữ bị méo hình học, dẫn đến rec lỗi.
  - Việc biến đổi hình học (rectification) là cần thiết để đưa vùng chữ về mặt phẳng Euclidean.

**Các bước thực hiện:**
1. **Input: Box 4 điểm (quadrilateral)**
Mỗi box là 1 polygon gồm 4 điểm: [x1, y1], [x2, y2], [x3, y3], [x4, y4], đi theo thứ tự top-left → clockwise.
Các điểm này biểu diễn 4 đỉnh của vùng chữ đã phát hiện (có thể nghiêng/lệch).
2. Tư duy hình học: từ tứ giác → hình chữ nhật phẳng
  - Một tứ giác trong ảnh là biểu diễn perspective projection của một vùng chữ nằm ngang.
  - Để khôi phục chữ về dạng "ngay ngắn", ta cần tìm một phép biến đổi hình học đưa 4 điểm này về hình chữ nhật phẳng.

Đây là bài toán đồng nhất perspective giữa 2 hệ tọa độ:
| Gốc ảnh                | Đích phẳng (chuẩn) |
| ---------------------- | ------------------ |
| `[x1, y1]` (top-left)  | `[0, 0]`           |
| `[x2, y2]` (top-right) | `[w - 1, 0]`       |
| `[x3, y3]` (bot-right) | `[w - 1, h - 1]`   |
| `[x4, y4]` (bot-left)  | `[0, h - 1]`       |

  - Với w, h là width/height thực tế của box, tính bằng độ dài cạnh.
  - Kết quả: vùng ảnh chữ được cắt ra, căn thẳng, không lệch trục.

3. Output: List các patch ảnh chứa chữ (rectified patches)
  - Mỗi patch có hình chữ nhật, kích thước tự do (tùy theo box).
  - Dùng cho bước tiếp theo: Resize + Normalize (Recognition Preprocessing)

**Notes & Implementation Details**
| Vấn đề thực tế                         | Hướng xử lý                                                               |
| -------------------------------------- | ------------------------------------------------------------------------- |
| Box bị méo hoặc thứ tự điểm sai        | Cần chuẩn hóa thứ tự điểm về **top-left → clockwise** trước khi transform |
| Box có kích thước quá nhỏ (e.g. < 5px) | Có thể bỏ qua do không đủ chi tiết cho rec                                |
| Ảnh bị mất nét sau crop                | Thường do box co lại quá mức từ threshold, hoặc thiếu bước `unclip`       |
| Hỗ trợ ảnh grayscale                   | Nên convert sang 3-channel (RGB) để thống nhất input                      |
| Border bị cắt cụt                      | Phải đảm bảo tọa độ box đã clip về trong ảnh gốc (không vượt biên)        |

**Tóm lại**
Text Region Cropping là bước chuyển đổi hình học quan trọng giữa Detection và Recognition.
Nếu box không được transform đúng:
  - Text bị nghiêng hoặc méo → Recognition model hiểu sai
  - Chữ bị cắt thiếu nét → rec ra chữ lỗi
Việc đảm bảo mỗi patch được perspective rectified là tiền đề sống còn cho độ chính xác của toàn pipeline.

### 2.3 Recognition Phase
#### 2.3.1 Recognition Preprocessing
**Mục tiêu** 
Biến mỗi text patch (sau crop) thành tensor phù hợp với model recognition, giữ nguyên nội dung, tỷ lệ, và format.

**Pipeline tổng thể**
[Text Patch Image]
   ↓
[Resize (Height = 48)]
   ↓
[Padding to max width (e.g. 320)]
   ↓
[Normalize pixel → [-1, 1]]
   ↓
[Reformat to Tensor: [1, 3, 48, W]]

##### 1. Resize to Standard height
**Mục tiêu**
Chuyển mỗi ảnh text patch (với kích thước tuỳ ý) thành ảnh có chiều cao cố định = 48, trong khi giữ nguyên tỷ lệ khung hình (aspect ratio).

**Vì sao phải resize về H = 48?**
Không phải vì “model yêu cầu” một cách máy móc, mà vì bản thân kiến trúc của các recognition model như **CRNN**, **SVTR**, **Rosetta** được xây dựng dựa trên assumptions sau:

###### 1.1 Text là sequence nằm ngang
- Text trong thực tế (scene text, printed text) chủ yếu là chuỗi ký tự nằm ngang, ít khi dọc
- Để tận dụng tính tuyến tính của ngôn ngữ → model cần ảnh có shape [height, width] với width tùy biến

###### 1.2 Chiều cao cố định giúp mô hình học tốt
- Recognition model có kiến trúc tổng quát:
  [Input Image] → [CNN Backbone] → [Feature Map] → [Sequence Encoder (BiLSTM/Transformer)] → [CTC/FC]
    - CNN backbone có nhiều tầng `stride = 2`, khiến chiều cao bị giảm dần qua từng tầng
    - Nếu chiều cao ban đầu quá nhỏ → sau khi downsample sẽ thành **1** → mất sạch hình dạng chữ
    - Nếu chiều cao quá lớn → model nặng, chậm, khó train

**Chiều cao giảm qua các tầng CNN (stride 2):**

| Tầng  | H input   | Stride | H feature  |
| ----- | --------- | ------ | ---------- |
| Input | 48        | –      | 48         |
| Conv1 | 48        | 2      | 24         |
| Conv2 | 24        | 2      | 12         |
| Conv3 | 12        | 2      | 6          |
| Conv4 | 6         | 2      | **3**      |

- Với `H input = 48`, ta thu được `H feature = 3`  
- Đây là mức tối thiểu để mô hình vẫn giữ được **các đặc trưng hình học dọc**

**Ghi chú**
- Nếu input cao hơn hoặc thấp hơn → các stroke chữ bị nát hoặc mất nét
- Cụ thể:
  - Vì ta encode ảnh thành sequence → mỗi column của feature map là 1 vector đại diện cho 1 "rãnh dọc" trên ảnh chữ (ví dụ: vector đại diện cho nét dọc chữ “b”, “h”, “i”, v.v.)
  - Nếu H feature < 3:
    - Mất nét dọc, các chữ có phần thẳng đứng bị biến mất (chữ “i” sẽ thành dấu chấm)
    - Không còn “hình dạng chữ” để rec
  - Nếu H feature = 1:
    - Vector đầu ra chỉ là trung bình toàn bộ chiều dọc → mất sạch cấu trúc chữ cái

**Tại sao không resize lên 64 hay 96?**
  - Dù tăng H input giúp tăng độ phân giải chiều dọc → nhưng feature map cũng sẽ to hơn, dẫn đến:
    - Model nặng hơn
    - Inference chậm hơn
    - Lãng phí nếu text không cần chi tiết cao
→ H input = 48 là tối ưu giữa chi tiết vs compute
- Thực nghiệm (trong cả paper và PaddleOCR config) cho thấy:
  - Với H input = 48
  → H_feature = 3
  → Mỗi vector output [B, C, 3, T] giữ được:
    - Stroke cao nhất (chữ dài như “h”, “b”)
    - Stroke thấp nhất (chữ ngắn như “o”, “e”)
    - Và trung tâm (chữ như “a”, “s”)
  → Ba điểm dọc là đủ để mô hình “nhìn ra” form chữ

**Resize như thế nào?**
- Giữ nguyên tỷ lệ khung hình (aspect ratio)
- Gọi h, w là chiều cao và rộng của patch
- Tính chiều rộng mới: w' = (w.48)/h
- Resize ảnh về (48, w')
→ Tránh biến dạng chữ (e.g. chữ “i” thành “I” do co dãn sai)

**Tóm tắt I/O**
| Input                    | Output                    |
|--------------------------|---------------------------|
| Text patch `[h, w, 3]`   | `[48, w', 3]`             |
| Condition `w'` có thể thay đổi theo tỷ lệ khung hình |
