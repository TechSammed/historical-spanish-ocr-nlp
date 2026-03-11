from utils.tesseract_line_extractor import extract_lines_from_page

extract_lines_from_page(
    image_path="data/images/train/page_12.png",  # change filename
    output_img_dir="data/train_lines",
    output_txt_dir="data/train_line_labels",
    page_id="page12"
)