import os
import cv2
import pytesseract
from pytesseract import Output


def extract_lines_from_page(image_path, output_img_dir, output_txt_dir, page_id, conf_threshold=40):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    image = cv2.imread(image_path)
    h_img, w_img, _ = image.shape

    data = pytesseract.image_to_data(
        image,
        lang="spa",
        output_type=Output.DICT
    )

    n_boxes = len(data["text"])

    lines = {}

    # Group words by (block, paragraph, line)
    for i in range(n_boxes):

        text = data["text"][i].strip()
        conf = int(data["conf"][i])

        if text == "" or conf < conf_threshold:
            continue

        block = data["block_num"][i]
        par = data["par_num"][i]
        line = data["line_num"][i]

        key = (block, par, line)

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )

        if key not in lines:
            lines[key] = {
                "words": [],
                "coords": [x, y, x + w, y + h]
            }
        else:
            # Expand bounding box
            lines[key]["coords"][0] = min(lines[key]["coords"][0], x)
            lines[key]["coords"][1] = min(lines[key]["coords"][1], y)
            lines[key]["coords"][2] = max(lines[key]["coords"][2], x + w)
            lines[key]["coords"][3] = max(lines[key]["coords"][3], y + h)

        lines[key]["words"].append(text)

    line_counter = 0

    for key in sorted(lines.keys()):
        x1, y1, x2, y2 = lines[key]["coords"]

        # Add padding
        pad = 8
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w_img, x2 + pad)
        y2 = min(h_img, y2 + pad)

        line_img = image[y1:y2, x1:x2]

        line_text = " ".join(lines[key]["words"]).strip()

        if len(line_text) < 2:
            continue

        img_name = f"{page_id}_line_{line_counter}.png"
        txt_name = f"{page_id}_line_{line_counter}.txt"

        cv2.imwrite(os.path.join(output_img_dir, img_name), line_img)

        with open(os.path.join(output_txt_dir, txt_name), "w", encoding="utf-8") as f:
            f.write(line_text)

        line_counter += 1

    print(f"{page_id}: Extracted {line_counter} aligned lines")