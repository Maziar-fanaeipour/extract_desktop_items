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
    """Detect and extract text using EasyOCR with English focus and validation."""
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
            logging.info(f"Raw detection: text={text}, confidence={confidence}")
            
            # Validate text contains mostly English characters
            if confidence > 0.3 and 10 < w < 400 and 5 < h < 150 and re.search('^[a-zA-Z0-9\s.,-()]+$', text):
                text_regions.append((text, x, y, w, h))
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save image with detected regions
        cv2.imwrite('text_regions.png', image)
        logging.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    except Exception as e:
        logging.error(f"Text detection error: {str(e)}")
        raise

def save_to_csv(texts_with_coords, output_path='output/desktop_items.csv'):
    """Save only text and coordinates to CSV with adjusted encoding."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data = [{'Name': text, 'X': x, 'Y': y, 'Width': w, 'Height': h} for text, x, y, w, h in texts_with_coords]
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')  # Changed to utf-8
        logging.info(f"Results saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"CSV saving error: {str(e)}")
        raise

def process_desktop_image(image_path):
    """Process desktop image and produce CSV with text only."""
    try:
        # Preprocess
        image = preprocess_image(image_path)
        
        # Detect and extract text
        texts_with_coords = detect_and_extract_text(image)
        
        # Save to CSV
        output_csv = save_to_csv(texts_with_coords)
        
        return output_csv
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        raise

if __name__ == "__main__":
    # Ensure EasyOCR is installed: pip install easyocr
    #image_path = 'desktop_image.png'
    image_path = 'Capture.PNG'
    try:
        output_csv = process_desktop_image(image_path)
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error: {str(e)}")