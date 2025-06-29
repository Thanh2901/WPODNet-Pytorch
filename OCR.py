import re
import sys
import cv2
import numpy as np
from paddleocr import PaddleOCR

def enhance_image(image_path):
    """
    Làm nét và cải thiện chất lượng ảnh trước khi OCR
    """
    print("Đang cải thiện chất lượng ảnh...")
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Tăng kích thước ảnh lên đáng kể để OCR dễ đọc hơn
    height, width = gray.shape
    scale_factor = max(3.0, 200/height)  # Tăng lên ít nhất 3 lần hoặc đến độ cao 200px
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(f"Đã tăng kích thước từ {width}x{height} lên {new_width}x{new_height}")
    
    # 2. Khử nhiễu
    denoised = cv2.fastNlMeansDenoising(resized)
    
    # 3. Cải thiện độ tương phản mạnh hơn
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 4. Áp dụng adaptive threshold để tách text khỏi background
    # Thử cả hai loại threshold
    thresh1 = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 5. Thử nghiệm với Otsu threshold
    _, otsu_thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Làm nét mạnh hơn
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel_sharp)
    
    # 7. Morphological operations để làm rõ ký tự
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    # Lưu các phiên bản khác nhau để test
    base_name = image_path.replace('.jpg', '').replace('.png', '')
    cv2.imwrite(f"{base_name}_enhanced_original.jpg", contrast_enhanced)
    cv2.imwrite(f"{base_name}_enhanced_thresh1.jpg", thresh1)
    cv2.imwrite(f"{base_name}_enhanced_thresh2.jpg", thresh2)
    cv2.imwrite(f"{base_name}_enhanced_otsu.jpg", otsu_thresh)
    cv2.imwrite(f"{base_name}_enhanced_sharp.jpg", sharpened)
    cv2.imwrite(f"{base_name}_enhanced_morph.jpg", opening)
    
    print(f"Đã lưu 6 phiên bản ảnh cải thiện khác nhau")
    
    # Trả về ảnh tốt nhất (thử với contrast enhanced trước)
    return [contrast_enhanced, thresh1, thresh2, otsu_thresh, sharpened, opening]

def main():
    # Kiểm tra có nhập đường dẫn ảnh không
    if len(sys.argv) != 2:
        print("Cách dùng: python OCR.py <đường_dẫn_ảnh>")
        print("Ví dụ: python OCR.py ./image.jpg")
        return
        
    image_path = sys.argv[1]
    
    # Cải thiện chất lượng ảnh trước khi OCR
    enhanced_images = enhance_image(image_path)
    if enhanced_images is None:
        return
    
    # Khởi tạo OCR với cấu hình cơ bản nhất
    ocr = PaddleOCR(lang='en')
    
    # Thử OCR với tất cả các phiên bản ảnh đã cải thiện
    all_results = []
    image_names = ['contrast', 'thresh1', 'thresh2', 'otsu', 'sharpened', 'morphed']
    
    for i, enhanced_img in enumerate(enhanced_images):
        print(f"\nĐang thực hiện OCR với ảnh {image_names[i]}...")
        try:
            result = ocr.predict(enhanced_img)
            all_results.append((result, image_names[i]))
            print(f"OCR {image_names[i]} hoàn thành")
        except Exception as e:
            print(f"Lỗi OCR với {image_names[i]}: {e}")
            all_results.append((None, image_names[i]))
    
    # Xử lý tất cả kết quả
    all_license_plates = []
    all_texts_combined = []
    
    for result, img_name in all_results:
        if result and result[0]:
            print(f"\n--- Kết quả từ ảnh {img_name} ---")
            for line in result[0]:
                if len(line) >= 2:
                    text = line[1][0].strip()
                    confidence = line[1][1] if len(line[1]) > 1 else 0
                    
                    if confidence > 0.2:  # Giảm ngưỡng tin cậy
                        all_texts_combined.append(text)
                        print(f"Text: '{text}' (tin cậy: {confidence:.2f}) - từ {img_name}")
                        
                        # Tìm biển số trong text
                        text_clean = text.upper().replace(' ', '').replace('-', '').replace('.', '')
                        
                        # Thêm pattern đặc biệt cho biển số châu Âu
                        european_patterns = {
                            'european_1': re.compile(r'[A-Z]{1,2}[\s\-\.]*[A-Z]{2}[\s\-\.]*\d{3,4}'),  # M PE 3389
                            'european_2': re.compile(r'[A-Z]\d*[A-Z]{2}\d{4}'),  # MPE3389
                            'european_3': re.compile(r'[A-Z][\s\-\.]*[A-Z]{2}[\s\-\.]*\d{4}'),  # M.PE.3389
                        }
                        
                        for pattern_name, pattern in european_patterns.items():
                            matches = pattern.findall(text.upper())
                            for match in matches:
                                all_license_plates.append({
                                    'text': match,
                                    'type': pattern_name,
                                    'confidence': confidence,
                                    'original': text,
                                    'source': img_name
                                })
    
    # Xử lý với các pattern cũ
    patterns = {
        'vn_old': re.compile(r'\b\d{2}[A-Z]-?\d{4,5}\b'),
        'vn_new': re.compile(r'\b\d{2}[A-Z]-?\d{3}\.\d{2}\b'),
        'vn_motorbike': re.compile(r'\b\d{2}[A-Z]{1,2}-?\d{4,5}\b'),
        'international': re.compile(r'\b[A-Z]{1,3}\.?[A-Z]{2}\.?\d{3,4}\b'),
        'general': re.compile(r'\b[A-Z0-9]{5,10}\b'),
    }
    # Tìm trong tất cả text từ các ảnh
    for text in all_texts_combined:
        for pattern_name, pattern in patterns.items():
            matches = pattern.findall(text.upper())
            for match in matches:
                all_license_plates.append({
                    'text': match,
                    'type': pattern_name,
                    'confidence': 0.8,
                    'original': text,
                    'source': 'combined'
                })
    
    # Xử lý text được ghép lại
    combined_text = ''.join(all_texts_combined).upper().replace(' ', '')
    print(f"\nText tổng hợp từ tất cả ảnh: '{combined_text}'")
    
    # Tìm trong text ghép
    for pattern_name, pattern in patterns.items():
        matches = pattern.findall(combined_text)
        for match in matches:
            all_license_plates.append({
                'text': match,
                'type': pattern_name + '_combined',
                'confidence': 0.7,
                'original': combined_text,
                'source': 'all_combined'
            })
    
    # Loại bỏ trùng lặp và sắp xếp theo độ tin cậy
    unique_plates = {}
    for plate in all_license_plates:
        key = plate['text']
        if key not in unique_plates or unique_plates[key]['confidence'] < plate['confidence']:
            unique_plates[key] = plate
    
    sorted_plates = sorted(unique_plates.values(), key=lambda x: x['confidence'], reverse=True)
    
    # In kết quả
    print("\n" + "="*70)
    print(f"Tổng cộng tìm thấy {len(all_texts_combined)} đoạn text từ {len(enhanced_images)} ảnh")
    print("="*70)
    
    if sorted_plates:
        print(f"Tìm thấy {len(sorted_plates)} biển số có thể:")
        for i, plate in enumerate(sorted_plates, 1):
            print(f"{i}. {plate['text']} (loại: {plate['type']}, tin cậy: {plate['confidence']:.2f})")
            print(f"   Từ text gốc: '{plate['original']}' - Nguồn: {plate.get('source', 'unknown')}")
    else:
        print("Không tìm thấy biển số nào")
        print("\nTất cả text đã phát hiện được:")
        for i, text in enumerate(all_texts_combined, 1):
            print(f"{i}. '{text}'")
        
        print("\nGợi ý cải thiện:")
        print("- Kiểm tra các file ảnh đã tạo ra (_enhanced_*.jpg)")
        print("- Thử điều chỉnh độ sáng/tương phản của ảnh gốc")
        print("- Chụp ảnh từ góc độ thẳng hơn")
        print("- Tăng độ phân giải ảnh gốc")

if __name__ == "__main__":
    main()