import os
from utils.tesseract_line_extractor import extract_lines_from_page

# folder containing page images
input_folder = "data/images/train"

# folders to store line images and labels
output_img_dir = "data/train_lines"
output_txt_dir = "data/train_line_labels"

# create folders if they don't exist
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_txt_dir, exist_ok=True)

# process each page image
for image_name in sorted(os.listdir(input_folder)):

    if image_name.endswith(".png"):

        page_id = image_name.replace(".png", "")
        image_path = os.path.join(input_folder, image_name)

        print(f"Processing {page_id}...")

        extract_lines_from_page(
            image_path=image_path,
            output_img_dir=output_img_dir,
            output_txt_dir=output_txt_dir,
            page_id=page_id
        )

print("\nAll pages processed.")