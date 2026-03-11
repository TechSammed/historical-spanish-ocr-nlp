import os
import cv2
import numpy as np


def segment_lines(image_path, output_folder, page_id):
    os.makedirs(output_folder, exist_ok=True)

    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Binarize (text becomes white)
    _, thresh = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ==========================
    # COLUMN DETECTION
    # ==========================
    vertical_sum = np.sum(thresh, axis=0)

    vertical_sum = cv2.GaussianBlur(
        vertical_sum.reshape(1, -1), (1, 51), 0
    ).flatten()

    vertical_sum_norm = vertical_sum / np.max(vertical_sum)

    valley_indices = np.where(vertical_sum_norm < 0.1)[0]

    columns = [image]  # default single column

    if len(valley_indices) > 0:
        splits = np.split(
            valley_indices,
            np.where(np.diff(valley_indices) != 1)[0] + 1
        )

        largest_valley = max(splits, key=len)

        if len(largest_valley) > image.shape[1] * 0.05:
            split_index = int(np.mean(largest_valley))

            left = image[:, :split_index]
            right = image[:, split_index:]

            # Ensure both halves contain enough content
            if np.sum(left) > 10000 and np.sum(right) > 10000:
                columns = [left, right]

    # ==========================
    # LINE SEGMENTATION
    # ==========================

    line_counter = 0

    for col in columns:

        _, col_thresh = cv2.threshold(
            col, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        horizontal_sum = np.sum(col_thresh, axis=1)

        threshold = np.max(horizontal_sum) * 0.15

        start = None

        for i in range(len(horizontal_sum)):

            if horizontal_sum[i] > threshold:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start > 15:

                        # ===== Padding Adjustment =====
                        top_pad = 10      # protect ascenders (f, R, b, l)
                        bottom_pad = 6    # protect descenders (g, p, y)

                        top = max(0, start - top_pad)
                        bottom = min(col.shape[0], i + bottom_pad)

                        line_img = col[top:bottom, :]

                        # ===== Noise Filtering =====
                        if line_img.shape[1] < 50:
                            start = None
                            continue

                        if np.sum(line_img) < 5000:
                            start = None
                            continue

                        cv2.imwrite(
                            os.path.join(
                                output_folder,
                                f"{page_id}_line_{line_counter}.png"
                            ),
                            line_img
                        )
                        line_counter += 1

                    start = None

        # Catch final line at end of column
        if start is not None and len(horizontal_sum) - start > 15:

            top_pad = 10
            bottom_pad = 6

            top = max(0, start - top_pad)
            bottom = col.shape[0]

            line_img = col[top:bottom, :]

            if line_img.shape[1] >= 50 and np.sum(line_img) >= 5000:
                cv2.imwrite(
                    os.path.join(
                        output_folder,
                        f"{page_id}_line_{line_counter}.png"
                    ),
                    line_img
                )
                line_counter += 1

    print(f"{page_id}: Extracted {line_counter} lines")