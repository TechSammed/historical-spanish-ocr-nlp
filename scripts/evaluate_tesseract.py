from jiwer import cer, wer

# load ground truth
with open("data/ground_truth/page_1.txt", "r", encoding="utf-8") as f:
    gt = f.read().replace("\n", " ")

# load Tesseract prediction
with open("results/page_1_tesseract.txt", "r", encoding="utf-8") as f:
    pred = f.read().replace("\n", " ")

print("Tesseract CER:", cer(gt, pred))
print("Tesseract WER:", wer(gt, pred))