import cv2
import numpy as np

def get_rotate_crop_image(img, points):
    """
    Crop text region from image based on 4 corner points
    Extracted from crop.py
    """
    assert len(points) == 4, "shape of points must be 4*2"
    
    # Calculate crop dimensions
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), 
            np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), 
            np.linalg.norm(points[1] - points[2])
        )
    )
    
    # Define target corners for perspective transform
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ])
    
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    
    # Auto-rotate if text is vertical (height > 1.5 * width)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
        
    return dst_img

def crop_text_regions(img, boxes):
    """
    Crop all detected text regions from image
    """
    cropped_images = []
    
    for box in boxes:
        # Convert box to numpy array with shape (4, 2)
        if isinstance(box, list):
            points = np.array(box).reshape(4, 2).astype(np.float32)
        else:
            points = box.reshape(4, 2).astype(np.float32)
        
        # Crop text region
        crop_img = get_rotate_crop_image(img, points)
        cropped_images.append(crop_img)
    
    return cropped_images

def order_points_clockwise(pts):
    """
    Reorder 4 points to: top-left, top-right, bottom-right, bottom-left
    Giả định rằng box là hình tứ giác lồi không tự cắt nhau
    """
    # Sort theo y tăng (trên → dưới)
    pts = sorted(pts, key=lambda x: x[1])

    # Lấy 2 điểm trên đầu (top)
    top_two = sorted(pts[:2], key=lambda x: x[0])  # theo x tăng: trái → phải
    bottom_two = sorted(pts[2:], key=lambda x: x[0], reverse=True)  # phải → trái

    return np.array([top_two[0], top_two[1], bottom_two[0], bottom_two[1]], dtype=np.float32)

def sort_boxes_top_to_bottom_left_to_right(boxes, y_thresh=10):
    """
    Sắp xếp các box theo thứ tự đọc: trên xuống dưới, trái qua phải.
    - `y_thresh`: Ngưỡng chênh lệch Y để coi là cùng 1 dòng.
    """
    # Step 1: Sort theo Y rồi theo X
    boxes = sorted(boxes, key=lambda box: (min(pt[1] for pt in box), min(pt[0] for pt in box)))

    # Step 2: Gom các box vào dòng rồi sort trong dòng
    lines = []
    current_line = [boxes[0]]
    for b in boxes[1:]:
        y_current = min(pt[1] for pt in current_line[-1])
        y_b = min(pt[1] for pt in b)
        if abs(y_b - y_current) < y_thresh:
            current_line.append(b)
        else:
            lines.append(sorted(current_line, key=lambda box: min(pt[0] for pt in box)))
            current_line = [b]
    lines.append(sorted(current_line, key=lambda box: min(pt[0] for pt in current_line)))

    # Flatten về 1 list
    return [box for line in lines for box in line]
