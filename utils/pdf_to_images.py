from pdf2image import convert_from_path
import os


def pdf_to_images(input_pdf, output_folder, dpi=300):
    os.makedirs(output_folder, exist_ok=True)

    print(f"Converting {input_pdf}...")
    pages = convert_from_path(input_pdf, dpi=dpi)

    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i+1}.png")
        page.save(image_path, "PNG")

    print(f"Saved {len(pages)} images to {output_folder}\n")


if __name__ == "__main__":

    # Convert training PDF
    pdf_to_images(
        input_pdf="data/train_pdfs/p15_to_p40.pdf",
        output_folder="data/images/train"
    )

    # Convert evaluation PDF
    pdf_to_images(
        input_pdf="data/eval_pdfs/p7_to_p9.pdf",
        output_folder="data/images/eval"
    )