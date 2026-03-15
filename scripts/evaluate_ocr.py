from jiwer import cer, wer

# load ground truth
with open("data/ground_truth/page_1.txt", "r", encoding="utf-8") as f:
    gt = f.read()

# load predicted text
with open("results/page_1_predicted.txt", "r", encoding="utf-8") as f:
    pred = f.read()

# normalize text (remove line breaks)
gt = gt.replace("\n", " ")
pred = pred.replace("\n", " ")

print("Character Error Rate:", cer(gt, pred))
print("Word Error Rate:", wer(gt, pred))