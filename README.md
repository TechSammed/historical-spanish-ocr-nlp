# Historical Spanish OCR using CRNN + LLM Post-processing

This project implements a complete OCR pipeline for early modern Spanish printed texts, focusing on improving recognition accuracy over traditional OCR systems like Tesseract using a deep learning-based CRNN model, along with LLM-based post-processing.

The pipeline processes scanned historical documents and performs text extraction, model training, inference, evaluation, and refinement using an LLM.

---

# Project Overview

Historical documents present unique OCR challenges:

- Irregular typography and fonts  
- Ink degradation and noise  
- Non-standard spelling conventions  
- Layout inconsistencies  

This project explores whether a custom-trained CRNN model + LLM refinement can outperform traditional OCR engines on such data.

---

# Complete OCR Pipeline

PDF → Image Conversion → Layout Cleaning → Line Segmentation  
→ Dataset Creation → CRNN Training → Inference  
→ Prediction Merging → LLM Post-processing  
→ Evaluation (CER/WER) → Baseline Comparison  

---

# 1. PDF → Image Conversion

PDF pages are converted into high-resolution images.

Configuration:
- Resolution: 300 DPI  
- Format: PNG  

Reason:
- CNN models require image input  
- 300 DPI balances clarity and computation cost  

Output:
data/images/

---

# 2. Baseline OCR (Tesseract)

Tesseract OCR (Spanish model) is used as the baseline system.

Observations:
- Works reasonably well on modern text  
- Struggles with:
  - historical fonts  
  - degraded scans  
  - irregular spacing  

---

# 3. Layout Cleaning

Margins were trimmed:
- Left: 2%  
- Right: 2%  
- Bottom: 4%  
- Top: 0%  

Result:
Minimal improvement → errors mainly due to character recognition, not layout.

---

# 4. Line Segmentation

Text lines extracted using pytesseract.image_to_data()

Example:
data/train_lines/page_1_line_0.png

---

# 5. Dataset Creation (Pseudo-labeling)

Each line image is paired with Tesseract output:

Line Image → OCR Text (pseudo-label)

Important:
This introduces label noise, affecting training quality.

---

# 6. Vocabulary Creation

Stored in:
models/vocab.json

Important:
Index 0 is reserved for CTC blank token

---

# 7. CRNN Model

Architecture:
CNN → Feature Extraction  
RNN → Sequence Modeling  
CTC → Decoding  

---

# 8. Training Process

Environment:
- Local + Google Colab (GPU)

Experiments:
- Initial: 1000 lines, 3 epochs  
- Extended: 3000 lines, 15 epochs  
- Final: 3000+ lines, 50 epochs  

Observations:
- Loss decreased  
- Model struggled with long sequences  
- Output often collapsed to:
  - single characters  
  - repeated tokens  

---

# 9. CRNN Inference

Implemented in:
utils/crnn_inference.py

Pipeline:
Image → Preprocess → Model → Argmax → CTC Decode  

---

# 10. Prediction Processing

Merged into page-level text:
results/page_1_predicted.txt

---

# 11. LLM-based Post-processing (Groq)

A Groq-hosted LLM (LLaMA 3.1) is used as a final refinement step.

Pipeline:
CRNN OCR → LLM Correction → Final Text

Approach:
- Text split into chunks to handle token limits  
- LLM constrained to:
  - only fix OCR errors  
  - avoid rewriting text  
  - preserve structure  

Key Insight:
LLMs improve readability but may not always improve CER/WER due to strict alignment metrics.

---

# 12. Evaluation Metrics

Character Error Rate (CER)  
Character-level edit distance  

Word Error Rate (WER)  
Word-level edit distance  

---

# 13. Evaluation Results

CRNN:
- CER: 26.70  
- WER: 32.51  

Tesseract:
- CER: 42.02  
- WER: 44.60  

CRNN + LLM:
- CER: 30.64  
- WER: 37.42  

---

# 14. Result Analysis

Observations:
- CRNN outperforms Tesseract  
- LLM slightly degrades CER/WER  
- However, LLM improves:
  - readability  
  - spelling consistency  

Key Insight:
LLMs improve qualitative output but may degrade strict OCR metrics.

---

# 15. Why Accuracy is Low

1. Noisy Labels  
Tesseract-generated labels introduce errors  

2. Limited Dataset  
Only ~3000 lines  

3. Historical Text Complexity  
Old fonts and spelling variations  

4. CTC Limitations  
Sequence alignment issues and output collapse  

5. Segmentation Errors  
Imperfect line crops  

---

# 16. Why This Approach Matters

- CRNN performs better than Tesseract  
- Demonstrates a complete OCR pipeline  
- Includes LLM integration  
- Reflects real-world research challenges  

---

# 17. Future Improvements

Data:
- Manual ground truth labels  
- Larger dataset (10k+ lines)

Model:
- Transformer-based OCR  
- Deeper CNN backbone  
- Beam search decoding  

LLM:
- Better prompt engineering  
- Alignment-aware correction  

Post-processing:
- Language models  
- Spell correction  

---

# 18. Project Structure

renai_project/

data/  
models/  
utils/  
scripts/  
results/  

---

# 19. Technologies Used

- Python  
- PyTorch  
- OpenCV  
- Tesseract OCR  
- pytesseract  
- Groq API (LLM)  
- Google Colab  

---

# 20. Key Learnings

- OCR pipelines are multi-stage systems  
- Data quality is more important than model complexity  
- Historical OCR is challenging  
- LLMs improve readability but not always metrics  
- Evaluation is critical  

---

# Final Conclusion

This project builds a complete OCR pipeline for historical Spanish texts, integrating deep learning OCR (CRNN), baseline comparison (Tesseract), and LLM-based refinement.

While accuracy remains limited due to dataset and label constraints, the CRNN model shows clear improvement over traditional OCR, and the integration of LLM demonstrates modern hybrid OCR approaches.