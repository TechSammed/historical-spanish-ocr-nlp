import cv2
import pytesseract
import os


def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    cv2.imwrite(output_path, blur)

    return image, blur


def run_ocr(image, language="spa"):
    text = pytesseract.image_to_string(image, lang=language)
    return text


if __name__ == "__main__":
    raw_image_path = "data/images/page_1.png"
    processed_image_path = "data/processed_images/page_1_processed.png"

    # Ensure processed_images folder exists
    os.makedirs("data/processed_images", exist_ok=True)

    # Preprocess
    raw_image, processed_image = preprocess_image(raw_image_path, processed_image_path)

    print("Running OCR on raw image...\n")
    raw_text = run_ocr(raw_image)
    print(raw_text[:500])  # print first 500 characters

    print("\nRunning OCR on processed image...\n")
    processed_text = run_ocr(processed_image)
    print(processed_text[:500])