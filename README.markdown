# Desktop Text Extraction Project

## Overview
This project is designed to extract text from desktop screenshots, particularly focusing on identifying and processing text labels of files, folders, and applications. It uses computer vision techniques and the EasyOCR library to detect and recognize text, even in complex backgrounds.

## Prerequisites
- Python 3.7 or higher
- Required libraries:
  - `opencv-python`
  - `easyocr`
  - `pandas`
  - `numpy`
- Install dependencies using:
  ```bash
  pip install opencv-python easyocr pandas numpy
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install the required dependencies as listed above.

## Usage
1. Place your desktop screenshot in the project directory and name it `desktop_image.png`.
2. Run the script:
   ```bash
   python multiline_easyocr_extract.py
   ```
3. Check the output in the `output/desktop_items.csv` file, which contains extracted text with their coordinates.
4. Review the intermediate images (`preprocessed_image.png` and `text_regions.png`) for debugging.

## Output
The script generates a CSV file (`output/desktop_items.csv`) with columns:
- `Name`: The extracted text
- `X`, `Y`: Coordinates of the top-left corner
- `Width`, `Height`: Dimensions of the text region

## Notes
- The script is optimized for English text on desktop screenshots.
- Multi-line text (e.g., "Docker Desktop") is merged into a single entry.
- Adjust the `merge_multiline_text` function parameters (e.g., vertical threshold) if needed for your images.

## Contributing
Feel free to submit issues or pull requests. Please ensure to follow the existing code style and include tests if applicable.

## License
[Specify your license, e.g., MIT, GPL, or none if not applicable]

## Contact
For questions or support, please open an issue in the repository or contact [your email or username].