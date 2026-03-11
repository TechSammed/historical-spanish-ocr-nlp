import pytesseract
import os
from PIL import Image


def run_baseline_ocr(image_folder, output_folder, language="spa"):
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(os.listdir(image_folder))

    for image_name in image_files:
        if image_name.endswith(".png"):
            image_path = os.path.join(image_folder, image_name)

            print(f"Running OCR on {image_name}...")
            image = Image.open(image_path)

            text = pytesseract.image_to_string(image, lang=language)

            output_file = os.path.join(
                output_folder,
                image_name.replace(".png", ".txt")
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

    print("\nBaseline OCR completed.")


if __name__ == "__main__":
    run_baseline_ocr(
        image_folder="data/images/train",
        output_folder="data/ground_truth/train_pseudo"
    )