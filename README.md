
```markdown
# Historical Spanish OCR using CRNN

This project focuses on building an OCR pipeline for **early modern Spanish printed texts**.  
The goal is to improve recognition accuracy beyond the baseline produced by traditional OCR engines like Tesseract by training a **custom CNN-RNN (CRNN) model**.

The system processes scanned historical documents and predicts text using deep learning-based sequence recognition.

---

# Project Pipeline

The project follows a multi-stage OCR pipeline:

1. PDF → Image Conversion  
2. Baseline OCR Evaluation  
3. Layout Cleaning  
4. CRNN Training Dataset Creation  
5. CRNN Model Training  
6. CRNN Inference Pipeline  

---

# 1. PDF → Image Conversion

PDF pages are rendered into high-resolution PNG images before processing.

### Why conversion is required

CNN-based models require images as input, while PDFs contain vector-based page representations.

### Resolution choice

| DPI | Reason |
|----|------|
150 DPI | Too low for historical fonts |
300 DPI | Balanced quality and speed |
600 DPI | Very heavy processing |

Final choice: **300 DPI**

Images are saved into:

```

data/images/

```

---

# 2. Image Preprocessing Study

Initial experiments evaluated whether preprocessing improves OCR performance.

Tested preprocessing steps:

- Grayscale conversion  
- Thresholding (binarization)  
- Denoising  

However, experiments showed that heavy preprocessing **degraded OCR performance**.

### Reason

Tesseract already performs:

- grayscale conversion  
- thresholding  
- layout detection  
- line segmentation  

Overprocessing distorted thin historical characters.

### Final decision

For the baseline:

```

Use raw 300 DPI images

```

---

# 3. Baseline OCR Evaluation

The **Spanish Tesseract OCR engine** was used to create a baseline.

Evaluation used **3 manually aligned pages with ground truth transcription**.

### Historical Spanish Normalization

Before computing metrics, normalization rules were applied:

- u and v treated as interchangeable  
- long s normalized (f → s)  
- ç converted to z  
- accents removed (except ñ)  
- whitespace normalized  

---

### Evaluation Metrics

Two standard OCR metrics were used.

**Character Error Rate (CER)**  
Measures character-level edit distance.

**Word Error Rate (WER)**  
Measures word-level edit distance.

---

### Baseline Results

| Metric | Score |
|------|------|
CER | ~10.5% |
WER | ~48.3% |

CER indicates moderate character accuracy, while high WER is expected due to:

- historical spelling variation  
- segmentation issues  
- hyphenation artifacts  

This baseline serves as the reference point for evaluating the CRNN model.

---

# 4. Layout Cleaning

To remove margin artifacts:

- 2% left trim  
- 2% right trim  
- 4% bottom trim  
- no top trim  

After cleaning, OCR evaluation was repeated.

### Clean Baseline Results

| Metric | Score |
|------|------|
CER | 10.7% |
WER | 48.3% |

Layout cleaning did **not significantly affect accuracy**, confirming that most OCR errors arise from **character confusions rather than layout noise**.

---

# 5. CRNN Training Data Preparation

Training data was generated automatically using **Tesseract line detection**.

### Line Segmentation

Using:

```

pytesseract.image_to_data()

```

Each detected text line was cropped and stored as an individual image.

Example:

```

data/train_lines/page12_line_0.png

```

Corresponding OCR text is stored as a pseudo-label:

```

data/train_line_labels/page12_line_0.txt

```

Each training sample becomes:

```

Line Image → Text Label

```

---

# 6. Vocabulary Creation

All characters appearing in pseudo-label files were collected to create a vocabulary.

Stored in:

```

models/vocab.json

```

Important rule:

```

Index 0 → reserved for CTC blank token

```

---

# 7. Dataset Pipeline

A custom **PyTorch dataset** was implemented.

### LineDataset

Loads:

- line images  
- text labels  

### Image Processing

Images are:

- converted to grayscale  
- resized to **64 × 256**  
- normalized  
- converted to tensors  

### Text Encoding

Text labels are encoded into integer sequences using:

```

TextEncoder

```

---

# Dataset Verification

Example DataLoader batch:

```

Images: torch.Size([4, 1, 64, 256])
Labels: torch.Size([128])
Lengths: tensor([45, 21, 32, 30])

```

This confirms correct batching for **CTC training**.

---

# 8. Initial CRNN Training

The CRNN model was first trained using a small subset of the dataset.

Training configuration:

```

Dataset size: 1000 line images
Epochs: 3
Batch size: 2
Optimizer: Adam
Loss: CTC Loss

```

Training results:

```

Epoch 1/3 Loss: 3.5465
Epoch 2/3 Loss: 3.0788
Epoch 3/3 Loss: 2.9995

```

The model was successfully saved as:

```

models/crnn_model.pth

```

---

# 9. CRNN Inference Pipeline

Inference utilities were implemented in:

```

utils/crnn_inference.py

```

Core components:

```

load_model()
preprocess_image()
decode_prediction()
predict_line()

```

### CTC Decoding Rules

- index **0 reserved for blank token**
- repeated characters collapsed
- indices mapped back to characters using vocabulary

---

# Extended Training Experiments

To improve recognition accuracy, additional training experiments were conducted.

### Experiment Setup

```

Training lines: 3000
Epochs: 15
Batch size: 2
Learning rate: 0.0005

```

Training successfully completed and the updated model was saved.

However, during inference testing the model predictions remained incorrect.

Example outputs:

```

Input line: "tesoro de la lengua"

Predicted outputs:
r:
v:
e
A e

```

The model frequently predicted **single characters or repetitive outputs** rather than full words.

---

# Observed Model Issues

The trained CRNN model currently struggles with:

- predicting **complete sequences**
- learning long text patterns
- generalizing across line images

Most predictions collapse into **single-character outputs**.

This behavior indicates possible issues such as:

- insufficient training data
- noisy pseudo-labels from Tesseract
- CTC training instability
- model underfitting

---

# Ongoing Work

Current experiments aim to improve model performance.

Active investigations include:

- increasing dataset size beyond **3000 lines**
- training for **50+ epochs**
- experimenting with **larger batch sizes**
- running training on **GPU using Google Colab**
- improving label quality
- testing different learning rates

These steps aim to reduce **Character Error Rate (CER)** compared to the Tesseract baseline.

---

# Next Steps

Planned improvements:

- train CRNN on larger dataset
- improve pseudo-label quality
- experiment with deeper CNN backbones
- implement beam search decoding
- integrate **LLM-based OCR post-correction**

---

# Project Structure

```

renai_project/

data/
images/
train_lines/
train_line_labels/

models/
crnn.py
vocab.json
crnn_model.pth

utils/
dataset.py
text_encoder.py
crnn_inference.py

train_crnn.py

```

---

# Technologies Used

- Python  
- PyTorch  
- Tesseract OCR  
- OpenCV  
- pytesseract  

---

# Goal

Develop an OCR system specialized for **historical Spanish texts**, improving recognition accuracy beyond standard OCR engines.
```
=======
# Historical Spanish OCR + NLP Pipeline

This project develops an OCR and Natural Language Processing pipeline for extracting and analyzing text from historical Spanish book scans.

## Objectives
- Extract text from scanned historical documents using OCR
- Perform preprocessing and normalization of OCR outputs
- Apply NLP techniques such as tokenization and Named Entity Recognition (NER)
- Enable lexical extraction and analysis from historical Spanish corpora

## Methods
- OCR-based text extraction from scanned pages
- Text preprocessing and normalization
- Tokenization and Named Entity Recognition using SpaCy
- Exploration of multilingual transformer models (mBERT) for semantic analysis

## Tools
Python, SpaCy, NLTK, Tesseract/EasyOCR, Transformers

## Project Status
Initial implementation completed (over 50% of the pipeline developed). Remaining components and full codebase will be uploaded soon.
>>>>>>> 9c6d8b5bde7ef2b36a5f3c699495764166473413
