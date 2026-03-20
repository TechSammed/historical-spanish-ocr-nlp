from jiwer import cer, wer

# Load ground truth
with open("data/ground_truth/page_1.txt", "r", encoding="utf-8") as f:
    gt = f.read()

# Load LLM output
with open("results/page_1_llm.txt", "r", encoding="utf-8") as f:
    pred = f.read()

# Remove line breaks
gt = gt.replace("\n", " ")
pred = pred.replace("\n", " ")

# 🔥 Add normalization HERE
def normalize(text):
    text = text.lower()
    text = text.replace("v", "u")
    text = text.replace("f", "s")
    return text

gt = normalize(gt)
pred = normalize(pred)

# Compute metrics
cer_score = cer(gt, pred)
wer_score = wer(gt, pred)

print("\nLLM Evaluation Results:")
print(f"Character Error Rate (CER): {cer_score}")
print(f"Word Error Rate (WER): {wer_score}")