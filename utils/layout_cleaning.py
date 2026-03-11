import os
import cv2


def clean_layout(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_name in sorted(os.listdir(input_folder)):
        if image_name.endswith(".png"):

            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)

            if image is None:
                continue

            h, w = image.shape[:2]

            # ---- Stable minimal trimming ----
            bottom_trim_ratio = 0.96   # removes bottom 6%
            left_trim_ratio = 0.02     # removes left 2%
            right_trim_ratio = 0.98    # removes right 2%

            bottom = int(bottom_trim_ratio * h)
            left = int(left_trim_ratio * w)
            right = int(right_trim_ratio * w)

            cropped = image[0:bottom, left:right]

            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, cropped)

            print(f"Cleaned {image_name}")

    print("\nMinimal layout cleaning completed.")


if __name__ == "__main__":
    clean_layout(
        input_folder="data/images/eval",
        output_folder="data/images/eval_clean"
    )