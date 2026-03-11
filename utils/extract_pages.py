from PyPDF2 import PdfReader, PdfWriter
import os


def extract_range(input_pdf, output_pdf, start_page, end_page):
    """
    Extract pages from start_page to end_page (inclusive).
    Page numbers are human-readable (starting from 1).
    """
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pages}")

    # Convert to 0-based indexing
    start_index = start_page - 1
    end_index = end_page - 1

    if end_index >= total_pages:
        raise ValueError("End page exceeds total number of pages.")

    for page_num in range(start_index, end_index + 1):
        writer.add_page(reader.pages[page_num])

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"Extracted pages {start_page} to {end_page} → {output_pdf}")


if __name__ == "__main__":

    input_pdf = "data/raw_pdfs/Covarrubias - Tesoro lengua.pdf" 

    # 🔹 1️⃣ Evaluation Set (Ground Truth Available)
    extract_range(
        input_pdf=input_pdf,
        output_pdf="data/eval_pdfs/p7_to_p9.pdf",
        start_page=7,
        end_page=9
    )

    # 🔹 2️⃣ Training Set (No Ground Truth Required)
    extract_range(
        input_pdf=input_pdf,
        output_pdf="data/train_pdfs/p15_to_p40.pdf",
        start_page=15,
        end_page=40
    )