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
Trước khi đưa ảnh vào model ONNX, ảnh cần được biến đổi về format và thống nhất scale để khớp với mô hình đã training.
Các bước cụ thể: 
1. Resize ảnh về kích thước cố định [640, 640]

✅ What:
Chuyển ảnh về kích thước cố định 640x640 pixel, bất kể input ban đầu là gì.

❓ Why – Lý do sâu:

PP-OCRv5_mobile_det sử dụng:
Backbone: PPLCNetV3
Detection Head: DB (Differentiable Binarization)
Ref: https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml

a. Kiến trúc PP-OCRv5_mobile_det yêu cầu ảnh đầu vào cố định [3, 640, 640]:

Các layer như Conv2D, DepthwiseConv, BatchNorm có weight được training theo kích thước này, và ONNX export đã cố định input shape.
    Khi export sang ONNX (hoặc static inference engine), toàn bộ kernel shape, stride, padding, input/output tensor shape được hard-code.
    Nếu bạn đưa ảnh kích thước khác vào:
    Layer Conv2D không matching shape → ONNX runtime báo lỗi.

Hoặc model “chạy được” nhưng output feature map bị lệch tầng → DB Head decode sai vùng chữ.
Nếu đưa input sai kích thước, model sẽ báo lỗi shape mismatch, hoặc tạo ra output DB map sai tỉ lệ với ảnh gốc.
Ngoài ra, postprocess (decode box) phụ thuộc vào tỷ lệ giữa ảnh và DB map, nên nếu shape lệch sẽ gây lỗi hoặc kết quả sai hoàn toàn.

📌 b. DB Head phụ thuộc vào tỷ lệ không gian giữa ảnh và output map
DB head không trực tiếp predict bounding box, mà sinh ra các map nhị phân:
    Binary map (text vs background)
    Threshold map
    Approximate binarized map
Các map này có shape cố định, ví dụ [160 × 160] (do backbone stride = 4)
Nếu ảnh input không đúng [640 × 640] thì:
Mỗi pixel trên map không còn tương ứng chính xác với vùng ảnh gốc
→ Decode box bị sai vị trí và scale
🧠 Do đó, resize đúng shape là bắt buộc để đảm bảo DB map phản ánh chính xác không gian ảnh gốc.

📌 c. Khác với Recognition, ở bước Detection không cần giữ nguyên aspect ratio khi resize ảnh

Việc resize trực tiếp thay vì padding giữ tỉ lệ là một lựa chọn thiết kế trong PaddleOCR vì:
🔄 Detection hoạt động ở cấp độ toàn ảnh (global layout), chứ không cần độ chính xác pixel-level như recognition. Khi resize méo, các đoạn văn bản vẫn giữ được tương quan không gian đủ để model nhận biết vùng có chữ.
🧠 Kiến trúc DB head không phụ thuộc tuyệt đối vào aspect ratio. Nó học dựa trên hình dạng vùng liên kết (connected region) hơn là chi tiết kích thước chính xác của từng ký tự.
⚡ Padding giữ tỉ lệ tuy giúp tránh méo hình, nhưng làm chậm inference:
      Gây thêm thao tác padding/tracking padding size.
      Cần xử lý ngược padding sau khi decode box.
      Phức tạp hơn nếu chạy batch-size >1 với nhiều tỉ lệ ảnh khác nhau.
✅ PaddleOCR chấp nhận trade-off: một mức méo nhẹ vẫn đảm bảo detect đủ tốt với đa số văn bản thật, trong khi giúp tăng tốc đáng kể cho inference.

2. Convert sang float32 (nếu ảnh là uint8)

✅ What:
Chuyển kiểu dữ liệu từ uint8 (ảnh đầu vào từ OpenCV) sang float32 — định dạng mà mô hình yêu cầu.

❓ Why – Lý do sâu:

a. ONNX Runtime chỉ chấp nhận input kiểu float32

Mô hình được huấn luyện và export với các tensor float32.
Nếu đưa vào uint8, ONNX Runtime sẽ:
    Báo lỗi không khớp kiểu
    Hoặc cast ngầm → dễ gây bug hoặc cho kết quả sai

b. Ép kiểu float32 là tiền đề bắt buộc trước khi normalize

Việc normalize sau đó (img / 255.0, trừ mean, chia std) yêu cầu input là float32.
Nếu thực hiện trên uint8:
    Kết quả phép chia có thể trả về float64 (gây lỗi khi đưa vào model)
    Hoặc chia sai do phép toán nguyên → ra toàn số 0

3. Chuẩn hóa bằng ImageNet mean/std

mean = [0.485, 0.456, 0.406]  
std  = [0.229, 0.224, 0.225]

Ref for mean and std: https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml

2 bước chuẩn hóa ảnh đầu vào:
Scale pixel từ [0, 255] → [0.0, 1.0]
Ref: https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.Normalize
Normalize ảnh bằng cách trừ mean và chia std của ImageNet, nhằm đưa pixel đầu vào về phân phối có mean ≈ 0 và std ≈ 1 trên từng channel, đúng như mô hình đã được pretrain.

❓ Why – Lý do sâu:

a. Backbone (PPLCNetV3) được pretrain trên ImageNet:
Các trọng số layer (conv, bn, relu) trong PPLCNetV3 được huấn luyện với input có mean/std như trên.
Nếu không chuẩn hóa đúng, input sẽ có phân phối khác →
    Feature map bị lệch toàn diện
    Các filter đã học từ dữ liệu gốc (ImageNet) không còn khớp
→ Giống như đưa ảnh “nhiễu sáng” hoặc “ngược màu” vào model → mô hình phản ứng sai hoặc cho kết quả rác.

b. Normalize giúp loại bỏ nhiễu ánh sáng và độ tương phản
Ảnh gốc có thể bị tối/sáng, nhiễu, độ tương phản cao thấp không ổn định
Việc normalize giúp:
Mỗi pixel mang thông tin tương đối, không tuyệt đối
Mô hình tập trung vào biên, cạnh, hình khối (shape) — thứ mà DB head cần để phân biệt vùng có chữ hay không

c. Tránh sai lệch số học và tăng ổn định khi inference

Giá trị pixel nhỏ (≈ ±1) sau normalize giúp tránh:
Overflow trong tính toán float
Gradient explode/vanish (nếu dùng backward debug)
Sai lệch hậu xử lý box nếu scale ảnh bị lệch

4. Chuyển ảnh từ [H, W, C] → [C, H, W]

✅ What:
Đổi thứ tự chiều dữ liệu ảnh từ format mặc định của OpenCV ([H, W, C]) sang format chuẩn tensor [C, H, W] mà model yêu cầu.

❓ Why – Lý do sâu:

a. Hầu hết framework deep learning (Paddle, PyTorch, ONNX) đều expect input tensor ở dạng:

[N, C, H, W]
(với N: batch size, C: số channel, H, W: chiều cao & chiều rộng)

b. Vì sao Conv2D cần channel C đứng đầu?
Các lớp convolution (Conv2D) hoạt động theo cấu trúc:
For each channel c:
    Output += Input[c] * Kernel[c]
Việc đưa channel lên đầu giúp framework:
    Truy cập kênh hiệu quả hơn trong memory (data locality tốt hơn)
    Dễ dàng chia tách per-channel filter khi optimize
    Hỗ trợ batch operation qua chiều N (batch) phía trước

c. Nếu giữ nguyên [H, W, C] → ONNX sẽ lỗi ngay
Conv2D layer đầu tiên sẽ expect input shape [1, 3, 640, 640]
Nếu bạn đưa [1, 640, 640, 3] → ONNX Runtime báo lỗi shape mismatch

d. Ngoài ra, một số backend inference không tự báo lỗi rõ
Với TensorRT, TVM hoặc custom engine: nếu không reshape đúng [C, H, W], bạn có thể bị:
    Silent failure: ảnh bị swap màu (RGB ↔ BGR)
    Output rác nhưng không lỗi
    Debug khó vì không biết do format hay model

5. Thêm batch dimension

✅ What:
Thêm một chiều ở đầu tensor để chuyển ảnh từ [C, H, W] → [1, C, H, W] (batch size = 1).

❓ Why – Lý do sâu:

a. ONNX model yêu cầu input có batch dimension:
Các mô hình ONNX, bao gồm PP-OCRv5_mobile_det, luôn khai báo input với shape [N, C, H, W]
Nếu bạn đưa ảnh thiếu batch dimension ([3, 640, 640]), ONNX Runtime sẽ:
    Báo lỗi Invalid input shape
    Hoặc ép reshape ngầm → gây ra bug ngầm, khó debug

b. Chuẩn bị cho batch inference:
Việc giữ cấu trúc batch-ready cho phép dễ dàng mở rộng về sau, chạy nhiều ảnh một lúc mà không cần refactor pipeline.

Input shape chính xác yêu cầu:

[1, 3, 640, 640]
    1 → batch size
    3 → RGB
    640 × 640 → spatial dimension

Nếu sai bất kỳ chiều nào:
Thiếu batch	-> NNX Runtime báo lỗi Invalid shape
Channel ≠ 3	-> Conv layer không khớp weight → lỗi hoặc output rác
Size ≠ 640x640 -> Output feature map sai → DB map sai → box sai

📌 Trong PaddleOCR, batch dimension được thêm tự động ở tầng `loader:`.  
Tuy nhiên, khi viết pipeline inference ONNX riêng, bạn **phải thêm thủ công** batch `[1, C, H, W]`, nếu không sẽ gặp lỗi shape.

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
