import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def preprocess_image(image_path):
    """Revert to balanced preprocessing for text detection."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding with original block size
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Edge detection with proven thresholds
        edges = cv2.Canny(thresh, 20, 80)
        
        # Dilate to connect text edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Save preprocessed image and edges for debugging
        cv2.imwrite('preprocessed_image.png', dilated)
        cv2.imwrite('edges_image.png', edges)
        return dilated, image
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
        raise

def detect_text_regions(preprocessed_image, original_image):
    """Detect text regions with balanced contour filtering."""
    try:
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Balanced filters to capture various text sizes
            if 15 < w < 300 and 8 < h < 100 and 0.5 < w/h < 20:
                # Exclude very large contours (likely icons)
                if w * h < 15000:
                    text_regions.append((x, y, w, h))
                    cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save image with detected regions
        cv2.imwrite('text_regions.png', original_image)
        logging.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    except Exception as e:
        logging.error(f"Text region detection error: {str(e)}")
        raise

def extract_text(image, text_regions):
    """Extract text from detected regions with enhanced OCR."""
    try:
        extracted_texts = []
        for (x, y, w, h) in text_regions:
            roi = image[y:y+h, x:x+w]
            pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            # Preprocess ROI to improve OCR
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # OCR with PSM 7 for single line text
            config = '--psm 7 --oem 3'
            text = pytesseract.image_to_string(pil_image, lang='fas+eng', config=config)
            text = text.strip()
            
            if text:
                logging.info(f"OCR result for region ({x}, {y}, {w}, {h}): {text}")
                extracted_texts.append((text, x, y, w, h))
        
        logging.info(f"Extracted {len(extracted_texts)} text items")
        return extracted_texts
    except Exception as e:
        logging.error(f"Text extraction error: {str(e)}")
        raise

def classify_text(texts_with_coords):
    """Classify texts as File, Folder, or Application."""
    try:
        categorized = []
        for text, x, y, w, h in texts_with_coords:
            item_type = 'Unknown'
            if text.endswith(('.exe', '.app')):
                item_type = 'Application'
            elif any(text.endswith(ext) for ext in ('.pdf', '.txt', '.docx', '.jpg', '.png')):
                item_type = 'File'
            elif '.' not in text or text.endswith('/'):
                item_type = 'Folder'
            
            categorized.append({
                'Name': text,
                'Type': item_type,
                'X': x,
                'Y': y,
                'Width': w,
                'Height': h
            })
        
        return categorized
    except Exception as e:
        logging.error(f"Text classification error: {str(e)}")
        raise

def save_to_csv(categorized_items, output_path='desktop_items.csv'):
    """Save results to CSV with proper UTF-8 encoding."""
    try:
        df = pd.DataFrame(categorized_items)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"Results saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"CSV saving error: {str(e)}")
        raise

def process_desktop_image(image_path):
    """Process desktop image and produce CSV."""
    try:
        # Preprocess
        preprocessed, original = preprocess_image(image_path)
        
        # Detect text regions
        text_regions = detect_text_regions(preprocessed, original)
        
        # Extract text
        texts_with_coords = extract_text(original, text_regions)
        
        # Classify
        categorized_items = classify_text(texts_with_coords)
        
        # Save to CSV
        output_csv = save_to_csv(categorized_items)
        
        return output_csv
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        raise

if __name__ == "__main__":
    image_path = 'desktop_image.png'
    try:
        output_csv = process_desktop_image(image_path)
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error: {str(e)}")