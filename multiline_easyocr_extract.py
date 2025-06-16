import cv2
import numpy as np
import easyocr
import pandas as pd
import logging
import os
import re

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def preprocess_image(image_path):
    """Minimize preprocessing to preserve text details."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply minimal blur to reduce noise without losing details
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Reduced kernel size
        
        # Save preprocessed image for debugging
        cv2.imwrite('preprocessed_image.png', blurred)
        return image
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
        raise

def detect_and_extract_text(image):
    """Detect and extract text using EasyOCR with English focus."""
    try:
        # Initialize EasyOCR reader for English only
        reader = easyocr.Reader(['en'])
        
        # Detect text
        results = reader.readtext(image)
        
        text_regions = []
        for detection in results:
            bbox, text, confidence = detection
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x, y = int(top_left[0]), int(top_left[1])
            w = int(bottom_right[0] - top_left[0])
            h = int(bottom_right[1] - top_left[1])
            
            # Log raw detection for debugging
            logging.info(f"Raw detection: text={text}, confidence={confidence}, x={x}, y={y}, w={w}, h={h}")
            
            # Validate text contains mostly English characters
            if confidence > 0.3 and 10 < w < 400 and 5 < h < 200 and re.search('^[a-zA-Z0-9\s()+-.,]+$', text):
                text_regions.append((text, x, y, w, h))
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save image with detected regions
        cv2.imwrite('text_regions.png', image)
        logging.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    except Exception as e:
        logging.error(f"Text detection error: {str(e)}")
        raise

def merge_multiline_text(text_regions):
    """Merge text regions that are vertically close and horizontally aligned."""
    if not text_regions:
        return text_regions
    
    merged_regions = []
    current_group = []
    sorted_regions = sorted(text_regions, key=lambda x: (x[1], x[0]))  # Sort by Y then X
    
    for i in range(len(sorted_regions)):
        text, x, y, w, h = sorted_regions[i]
        if not current_group:
            current_group.append((text, x, y, w, h))
            continue
        
        prev_text, prev_x, prev_y, prev_w, prev_h = current_group[-1]
        # Check if current text is close vertically and aligned horizontally
        if (y - (prev_y + prev_h) < 20) and (abs(x - prev_x) < 50):
            current_group.append((text, x, y, w, h))
        else:
            if len(current_group) > 1:
                merged_text = ' '.join(t[0] for t in current_group)
                merged_x = min(t[1] for t in current_group)
                merged_y = min(t[2] for t in current_group)
                merged_w = max(t[1] + t[3] for t in current_group) - merged_x
                merged_h = max(t[2] + t[4] for t in current_group) - merged_y
                merged_regions.append((merged_text, merged_x, merged_y, merged_w, merged_h))
            else:
                merged_regions.append(current_group[0])
            current_group = [(text, x, y, w, h)]
    
    # Handle the last group
    if len(current_group) > 1:
        merged_text = ' '.join(t[0] for t in current_group)
        merged_x = min(t[1] for t in current_group)
        merged_y = min(t[2] for t in current_group)
        merged_w = max(t[1] + t[3] for t in current_group) - merged_x
        merged_h = max(t[2] + t[4] for t in current_group) - merged_y
        merged_regions.append((merged_text, merged_x, merged_y, merged_w, merged_h))
    elif current_group:
        merged_regions.append(current_group[0])
    
    return merged_regions

def save_to_csv(texts_with_coords, output_path='output/desktop_items.csv'):
    """Save only text and coordinates to CSV with adjusted encoding."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [{'Name': text, 'X': x, 'Y': y, 'Width': w, 'Height': h} for text, x, y, w, h in texts_with_coords]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Results saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"CSV saving error: {str(e)}")
        raise

def process_desktop_image(image_path):
    """Process desktop image and produce CSV with merged multiline text."""
    try:
        # Preprocess
        image = preprocess_image(image_path)
        
        # Detect and extract text
        texts_with_coords = detect_and_extract_text(image)
        
        # Merge multiline text
        merged_texts = merge_multiline_text(texts_with_coords)
        
        # Save to CSV
        output_csv = save_to_csv(merged_texts)
        
        return output_csv
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        raise

if __name__ == "__main__":
    # Ensure EasyOCR is installed: pip install easyocr
    image_path = 'Capture.PNG'
    try:
        output_csv = process_desktop_image(image_path)
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error: {str(e)}")