
import torch
import json
from models.crnn import CRNN
import cv2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------
# Load trained model
# ----------------------------------------------------
def load_model():

    with open("models/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    checkpoint = torch.load("models/crnn_epoch_50.pth", map_location=DEVICE)

    num_classes = checkpoint["fc.weight"].shape[0]

    model = CRNN(num_classes)
    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully")
    print("Number of characters in vocab:", len(vocab))

    return model, vocab


# ----------------------------------------------------
# Image preprocessing
# ----------------------------------------------------
def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    # resize to match training input
    img = cv2.resize(img, (256, 64))

    # normalize pixels
    img = img / 255.0

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    return img.to(DEVICE)


# ----------------------------------------------------
# CTC greedy decoding
# ----------------------------------------------------
def decode_prediction(indices, vocab):

    inv_vocab = {v: k for k, v in vocab.items()}

    decoded_text = ""
    previous_index = None

    for index in indices:

        # skip blank token
        if index != 0 and index != previous_index:
            decoded_text += inv_vocab.get(index, "")

        previous_index = index

    return decoded_text


# ----------------------------------------------------
# Predict text from a single line image
# ----------------------------------------------------
def predict_line(model, vocab, image_path):

    img = preprocess_image(image_path)

    with torch.no_grad():
        preds = model(img)

    preds = torch.argmax(preds, dim=2)

    indices = preds[0].cpu().numpy()

    text = decode_prediction(indices, vocab)

    return text


# ----------------------------------------------------
# Run inference on multiple lines
# ----------------------------------------------------
def test_model(model, vocab, folder, num_samples=100):

    files = sorted(os.listdir(folder))[:num_samples]

    os.makedirs("results", exist_ok=True)

    output_path = "results/crnn_predictions.txt"

    with open(output_path, "w", encoding="utf-8") as f:

        print("\nTesting CRNN OCR predictions:\n")

        for file in files:

            path = os.path.join(folder, file)

            text = predict_line(model, vocab, path)

            line = f"{file} → {text}"

            print(line)

            f.write(line + "\n")

    print("\nPredictions saved to:", output_path)


# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":

    model, vocab = load_model()

    folder = "data/train_lines"

    test_model(model, vocab, folder, num_samples=30)
