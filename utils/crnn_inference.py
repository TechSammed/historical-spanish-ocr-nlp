import torch
import json
from models.crnn import CRNN
import cv2
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    with open("models/vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    checkpoint = torch.load("models/crnn_model1.pth", map_location=DEVICE)

    num_classes = checkpoint["fc.weight"].shape[0]

    model = CRNN(num_classes)

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model, vocab


def preprocess_image(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.resize(img, (256, 64))

    # NOTE: normalization removed temporarily to match training scale
    # img = img / 255.0

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    return img.to(DEVICE)


def decode_prediction(indices, vocab):

    # invert vocab: index → character
    inv_vocab = {v: k for k, v in vocab.items()}

    decoded_text = ""
    previous_index = None

    for index in indices:

        # skip CTC blank (index 0)
        # skip repeated characters
        if index != 0 and index != previous_index:
            decoded_text += inv_vocab.get(index, "")

        previous_index = index

    return decoded_text


def predict_line(model, vocab, image_path):

    model.eval()
    img = preprocess_image(image_path)

    with torch.no_grad():
        preds = model(img)

    print("Model output shape:", preds.shape)
    print("Max prediction value:", preds.max().item())

    # choose highest probability class
    preds = torch.argmax(preds, dim=2)

    # get sequence for first batch
    indices = preds[0].cpu().numpy()

    print("Predicted indices:", indices[:20])

    text = decode_prediction(indices, vocab)

    return text


if __name__ == "__main__":

    model, vocab = load_model()

    test_images = [
        "data/train_lines/page12_line_0.png",
        "data/train_lines/page12_line_1.png",
        "data/train_lines/page12_line_2.png"
    ]

    for img_path in test_images:

        text = predict_line(model, vocab, img_path)

        print(img_path, "→", text)