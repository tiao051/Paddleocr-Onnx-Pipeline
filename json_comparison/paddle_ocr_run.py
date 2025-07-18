from paddleocr import PaddleOCR
import os, json, time
import numpy as np
from PIL import Image
import cv2

# --- CẤU HÌNH ---
input_dir = r"D:\Sozoo_Studio\v5_model\onnx_model\images"
output_dir = r"D:\Sozoo_Studio\v5_model\onnx_model\paddle_output"
os.makedirs(output_dir, exist_ok=True)

img_file = "eng_test1.jpg"
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# --- KIỂM TRA TÊN FILE ---
if not img_file.lower().endswith(valid_exts):
    print(f"⚠️ Invalid file extension: {img_file}")
else:
    img_path = os.path.join(input_dir, img_file)

    # Nếu là webp → chuyển sang RGB JPG
    if img_file.lower().endswith('.webp'):
        image = Image.open(img_path).convert('RGB')
        img_path = os.path.join(input_dir, img_file + '_converted.jpg')
        image.save(img_path)

    # --- KHỞI TẠO PADDLEOCR ---
    ocr = PaddleOCR(
        use_textline_orientation=False, 
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False
    )  # lang='vi' nếu muốn

    # --- CHẠY INFERENCE ---
    start_time = time.time()
    result = ocr.predict(img_path)
    duration = time.time() - start_time

    # --- KIỂM TRA KẾT QUẢ ---
    if not result:
        print(f"⚠️ No OCR result for {img_file}")
        filtered_result = []
    else:
        filtered_result = []
        # PaddleOCR returns a list with one dictionary containing all results
        if isinstance(result, list) and len(result) > 0:
            page_data = result[0]  # Get first page data
            
            # Extract texts, scores, and boxes from the dictionary
            texts = page_data.get('rec_texts', [])
            scores = page_data.get('rec_scores', [])
            boxes = page_data.get('rec_polys', [])
            
            # Combine texts, scores, and boxes
            for i in range(len(texts)):
                try:
                    text = texts[i] if i < len(texts) else ""
                    score = scores[i] if i < len(scores) else 0.0
                    box = boxes[i] if i < len(boxes) else []
                    
                    filtered_result.append({
                        "box": np.array(box).tolist() if len(box) > 0 else [],
                        "text": text,
                        "score": score
                    })
                except Exception as e:
                    print(f"⚠️ Error processing item {i}: {e}")
                    continue
        else:
            print(f"⚠️ Unexpected result format: {type(result)}")
            print(f"Result content: {result}")

    # --- GHI FILE JSON ---
    out_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_result, f, ensure_ascii=False, indent=2)

    # --- VẼ KẾT QUẢ DETECTION/RECOGNITION ---
    if filtered_result:
        # Đọc hình ảnh gốc
        img_vis = cv2.imread(img_path)
        if img_vis is not None:
            # Vẽ từng box và text
            for i, item in enumerate(filtered_result):
                box = item['box']
                text = item['text']
                score = item['score']
                
                if len(box) == 4:  # Đảm bảo có 4 điểm
                    # Chuyển đổi box thành numpy array
                    box_np = np.array(box, dtype=np.int32)
                    
                    # Vẽ box (màu xanh lá)
                    cv2.polylines(img_vis, [box_np], True, (0, 255, 0), 2)
                    
                    # Vẽ text và score
                    text_label = f"{text} ({score:.2f})"
                    # Lấy tọa độ góc trái trên để đặt text
                    text_x, text_y = box_np[0]
                    
                    # Vẽ nền cho text (màu đen với độ trong suốt)
                    text_size = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(img_vis, (text_x, text_y - text_size[1] - 5), 
                                 (text_x + text_size[0], text_y), (0, 0, 0), -1)
                    
                    # Vẽ text (màu trắng)
                    cv2.putText(img_vis, text_label, (text_x, text_y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Lưu hình ảnh kết quả
            vis_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '_visualization.jpg')
            cv2.imwrite(vis_path, img_vis)
            print(f"📸 Visualization saved → {vis_path}")
        else:
            print("⚠️ Could not read image for visualization")

    print(f"✅ {img_file} processed in {duration:.2f}s → {out_path}")

print("🎉 Done single image test.")
