# Historical Spanish OCR using CRNN + LLM Post-processing

This project implements a complete OCR pipeline for **early modern Spanish printed texts**, combining a **deep learning-based CRNN model** with **LLM-based post-processing (Groq)** to improve recognition quality over traditional OCR systems like Tesseract.

The system performs **end-to-end processing**, including image preparation, model training, inference, evaluation, and intelligent refinement.

---

# Project Overview

Historical OCR is significantly more difficult than modern OCR due to:

* Irregular typography and degraded print quality
* Ink noise and scanning artifacts
* Non-standard historical spelling
* Complex layouts and inconsistent spacing

This project explores whether a **custom CRNN model + LLM refinement pipeline** can improve OCR performance and usability.

---

# Complete OCR Pipeline

PDF → Image Conversion → Layout Cleaning → Line Segmentation
→ Dataset Creation → CRNN Training → Inference
→ Prediction Merging → LLM Post-processing
→ Evaluation (CER/WER) → Baseline Comparison

---

# LLM-Enhanced Pipeline (Final Stage)

CRNN OCR → Text Chunking → Groq LLM → Corrected Output → Evaluation

### LLM Role

* Correct OCR errors at **character/word level**
* Improve readability without rewriting content

### LLM Strategy

* Chunk-based processing (handles token limits)
* Low temperature (deterministic output)
* Strict prompt constraints:

  * No rewriting
  * No content addition
  * Minimal corrections only

### When LLM Fails

* Extremely noisy OCR output → LLM cannot infer correct text
* Heavily distorted words → semantic correction fails
* Strict evaluation → improved text may still increase CER/WER

---

# 1. PDF → Image Conversion

* Resolution: **300 DPI**
* Format: PNG

Reason:

* Required for CNN input
* Preserves character clarity

Output:
data/images/

---

# 2. Baseline OCR (Tesseract)

Used as a reference system.

### Observations:

* Works well for modern text
* Fails on:

  * historical fonts
  * degraded scans
  * irregular spacing

---

# 3. Layout Cleaning

Margins trimmed:

* Left: 2%
* Right: 2%
* Bottom: 4%
* Top: 0%

Result:
Minimal improvement → errors mainly from **character recognition**, not layout.

---

# 4. Line Segmentation

Extracted using:
pytesseract.image_to_data()

Output:
data/train_lines/page_1_line_0.png

---

# 5. Dataset Creation (Pseudo-labeling)

Each line image is paired with Tesseract text:

Line Image → OCR Text

### Limitation:

* Introduces **label noise**
* Directly affects model accuracy

---

# 6. Vocabulary Creation

Stored in:
models/vocab.json

Important:
Index 0 reserved for CTC blank token

---

# 7. CRNN Model

Architecture:
CNN → Feature Extraction
RNN (BiLSTM) → Sequence Modeling
CTC → Decoding

---

# 8. Training Process

Environment:

* Local + Google Colab (GPU)

### Experiments:

| Stage    | Lines | Epochs |
| -------- | ----- | ------ |
| Initial  | 1000  | 3      |
| Extended | 3000  | 15     |
| Final    | 3000+ | 50     |

### Observations:

* Loss decreased
* Model struggled with long sequences
* Output collapsed to:

  * single characters
  * repeated tokens

---

# 9. CRNN Inference

File:
utils/crnn_inference.py

Pipeline:
Image → Preprocess → Model → Argmax → CTC Decode

---

# 10. Prediction Processing

Merged into page-level text:
results/page_1_predicted.txt

---

# 11. LLM Post-processing (Groq)

### Pipeline:

CRNN Output → Chunking → LLM Correction → Final Text

### Improvements:

* Fixes OCR spelling errors
* Improves readability
* Handles noisy outputs

### Limitation:

* May degrade CER/WER due to strict alignment metrics

---

# 12. Evaluation Metrics

| Metric | Description          |
| ------ | -------------------- |
| CER    | Character Error Rate |
| WER    | Word Error Rate      |

---

# 13. Evaluation Results

| Model      | CER   | WER   |
| ---------- | ----- | ----- |
| CRNN       | 26.70 | 32.51 |
| Tesseract  | 42.02 | 44.60 |
| CRNN + LLM | 30.64 | 37.42 |

---

# 14. Result Analysis

### Key Findings:

* CRNN significantly outperforms Tesseract
* LLM improves **readability**, but:

  * slightly increases CER/WER
* Shows trade-off between:

  * human readability
  * strict evaluation metrics

---

# 15. Why Model Fails (Detailed Analysis)

The CRNN model does not fail randomly — it fails under specific conditions:

### 1. Noisy Supervision (Pseudo-label Problem)

* Labels generated using Tesseract contain errors
* Model learns incorrect patterns
* Leads to unstable predictions

---

### 2. Low Data Regime

* Only ~3000 training lines
* Deep models require much more data

Result:

* poor generalization
* overfitting

---

### 3. CTC Loss Instability

CTC struggles when alignment is unclear.

Failure patterns:

* repeated characters
* missing words
* single-character outputs

---

### 4. Long Sequence Difficulty

* Long historical lines
* RNN fails to retain context

Results:

* incomplete words
* truncated sequences

---

### 5. Segmentation Errors

* Incorrect line cropping
* Missing or broken characters

Impact:

* invalid input → poor predictions

---

### 6. Historical Text Complexity

* spelling variations
* uncommon characters
* inconsistent typography

---

### 7. Domain Shift Problem

* Training on noisy OCR labels
* Testing on clean ground truth

Result:

* increased CER/WER

---

# Failure Conditions Summary

The model performs poorly when:

* input is noisy or mis-segmented
* labels are incorrect
* sequences are long
* dataset is small
* characters are ambiguous

---

# Key Insight

The main bottleneck is **data quality**, not model architecture.

---

# 16. Why This Approach Matters

* Demonstrates full OCR pipeline
* Shows deep learning vs traditional OCR comparison
* Includes LLM integration
* Reflects real-world research challenges

---

# 17. Future Improvements

### Data

* Manual annotations
* Larger dataset (10k+ lines)

### Model

* Transformer-based OCR
* Deeper CNN backbone
* Beam search decoding

### LLM

* Better prompt engineering
* Alignment-aware correction

---

# 18. Project Structure

renai_project/

├── data/ → images, train_lines, labels, ground_truth
├── models/ → CRNN model + weights
├── utils/ → dataset + inference logic
├── scripts/ → preprocessing, evaluation, LLM pipeline
├── results/ → predictions and metrics
├── requirements.txt
├── .env
└── README.md

---

# Detailed Project Structure
```
renai_project/

├── data/
│   ├── images/                # Converted PDF pages (300 DPI)
│   ├── train_lines/           # Cropped line images
│   ├── train_line_labels/     # Pseudo-label text
│   └── ground_truth/          # Ground truth text

├── models/
│   ├── crnn.py                # CRNN architecture
│   ├── vocab.json             # Character mapping
│   └── crnn_epoch_50.pth      # Trained model

├── utils/
│   ├── dataset.py             # Dataset loader
│   ├── text_encoder.py        # Text encoding
│   └── crnn_inference.py      # Inference pipeline

├── scripts/
│   ├── generate_train_lines.py
│   ├── create_vocab.py
│   ├── merge_predictions.py
│   ├── merge_tesseract_predictions.py
│   ├── evaluate_ocr.py
│   ├── evaluate_tesseract.py
│   ├── evaluate_llm.py
│   └── llm_postprocess.py

├── results/
│   ├── crnn_predictions.txt
│   ├── tesseract_predictions.txt
│   ├── page_1_predicted.txt
│   ├── page_1_tesseract.txt
│   ├── page_1_llm.txt
│   └── metrics.txt

```
---

# 19. Technologies Used

| Category         | Tools        |
| ---------------- | ------------ |
| Language         | Python       |
| Deep Learning    | PyTorch      |
| OCR              | Tesseract    |
| Image Processing | OpenCV       |
| LLM              | Groq API     |
| Training         | Google Colab |

---

# 20. Key Learnings

* OCR is a multi-stage pipeline problem
* Data quality > model complexity
* Historical OCR is significantly harder
* LLMs improve readability but not always metrics
* Evaluation is essential

---

# Final Conclusion

This project builds a **complete OCR system for historical Spanish texts**, integrating:

* Deep learning OCR (CRNN)
* Traditional baseline comparison (Tesseract)
* LLM-based post-processing

While accuracy is limited due to dataset and label constraints, the CRNN model shows clear improvement over traditional OCR, and the integration of LLM demonstrates modern hybrid OCR approaches and their trade-offs.
