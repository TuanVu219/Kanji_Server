# effnet_version_of_code.py
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import threading

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224   # giống training
TOPK = 5

# Path đến checkpoint và labels
CKPT_PATH = os.path.join(os.path.dirname(__file__), "best_effnet_ema.pth")
LABELS_JSON = os.path.join(os.path.dirname(__file__), "class_names.json")

# =========================================================
# LOAD LABELS
# =========================================================
with open(LABELS_JSON, "r", encoding="utf-8") as f:
    labels = json.load(f)

idx2label = {i: l for i, l in enumerate(labels)}
num_classes = len(labels)

# =========================================================
# BUILD EfficientNet-B3 MODEL
# =========================================================
def build_efficientnet_b3(num_classes, pretrained=True, device=None):
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    if device:
        model.to(device)
    return model

# =========================================================
# LOAD CHECKPOINT
# =========================================================
def load_checkpoint_to_model(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")
    print("🔁 Loading EMA checkpoint...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "ema" in ckpt:
        sd = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    # strip "module." nếu có
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v

    model.load_state_dict(new_sd, strict=True)
    model.to(device)
    model.eval()
    print("✅ EMA model loaded.")
    return model

# =========================================================
# MODEL SINGLETON (thread-safe)
# =========================================================
_model = None
_model_lock = threading.Lock()

def get_effnet_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("🔹 Loading EfficientNet-B3 model (only once)...")
                model = build_efficientnet_b3(num_classes=num_classes, pretrained=True, device=DEVICE)
                model = load_checkpoint_to_model(model, CKPT_PATH, DEVICE)
                _model = model
    return _model

# =========================================================
# TRANSFORM / PREPROCESS
# =========================================================
preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_canvas_image(img):
    """PIL.Image -> tensor [1,C,H,W] on device"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return preprocess_transform(img).unsqueeze(0).to(DEVICE)

# =========================================================
# RECOGNITION
# =========================================================
def recognize_char(img, k=TOPK):
    """Nhận diện ký tự và trả về top-k nhãn dự đoán"""
    model = get_effnet_model()
    x = preprocess_canvas_image(img)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k)
    results = [(idx2label[top_idxs[i].item()], float(top_probs[i].item())) for i in range(k)]
    return results  # list [(label, prob), ...]

# =========================================================
# MERGE BOXES
# =========================================================
def merge_boxes(boxes, min_dist=10):
    merged = []
    for (x, y, w, h) in sorted(boxes, key=lambda b: b[0]):
        if not merged:
            merged.append([x, y, w, h])
        else:
            px, py, pw, ph = merged[-1]
            dist = x - (px + pw)
            if dist < -10:  # overlap
                nx = min(x, px)
                ny = min(y, py)
                nw = max(x + w, px + pw) - nx
                nh = max(y + h, py + ph) - ny
                merged[-1] = [nx, ny, nw, nh]
            else:
                merged.append([x, y, w, h])
    return merged

def recognize_and_merge_boxes_exhaustive(gray, boxes, threshold=0.7, max_span=None, k=TOPK):
    n = len(boxes)
    merged_boxes = []
    i = 0
    cache = {}
    H, W = gray.shape[:2]

    while i < n:
        best_prob = -1
        best_box = None
        best_j = i
        best_preds = None

        nx, ny, nx2, ny2 = boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]
        j_limit = n if max_span is None else min(n, i + max_span)

        for j in range(i, j_limit):
            if j > i:
                bx, by, bw, bh = boxes[j]
                nx = min(nx, bx)
                ny = min(ny, by)
                nx2 = max(nx2, bx + bw)
                ny2 = max(ny2, by + bh)

            x1 = max(0, int(nx))
            y1 = max(0, int(ny))
            x2c = min(W, int(nx2))
            y2c = min(H, int(ny2))
            if x2c <= x1 or y2c <= y1:
                continue

            key = (x1, y1, x2c, y2c)
            if key in cache:
                preds = cache[key]
            else:
                crop = Image.fromarray(gray[y1:y2c, x1:x2c]).convert("RGB")
                preds = recognize_char(crop, k=k)
                cache[key] = preds

            top_label, top_p = preds[0]
            if top_p > best_prob:
                best_prob = top_p
                best_box = [x1, y1, x2c - x1, y2c - y1]
                best_j = j
                best_preds = preds

        if best_box is None:
            merged_boxes.append(([0, 0, 0, 0], [("?", 0.0)]))
            i += 1
        elif best_j == i:
            merged_boxes.append((best_box, best_preds))
            i += 1
        else:
            if best_prob >= threshold:
                merged_boxes.append((best_box, best_preds))
                i = best_j + 1
            else:
                merged_boxes.append((boxes[i], [("?", 0.0)]))
                i += 1

    return merged_boxes  # [(box, predictions)]

# =========================================================
# SEGMENT CHARACTERS
# =========================================================
def segment_characters_from_image(img, k=TOPK):
    gray = np.array(img.convert("L"))
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=0)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = merge_boxes(boxes, min_dist=5)
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = recognize_and_merge_boxes_exhaustive(gray, boxes, threshold=0.7, k=k)
    return merged  # [(box, predictions)]

# =========================================================
# STROKES -> IMAGE
# =========================================================
def strokes_to_image(strokes, canvas_size=600, filename="output.png"):
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for stroke in strokes:
        for i in range(1, len(stroke)):
            x1, y1 = stroke[i - 1]
            x2, y2 = stroke[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=25, lineType=cv2.LINE_AA)
    pil_img = Image.fromarray(img)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(folder_path, filename)
    pil_img.save(save_path)
    print(f"✅ Image saved to {save_path}")
    return Image.fromarray(img)

# =========================================================
# HỖ TRỢ KIỂM TRA KANJI
# =========================================================
def is_kanji(word):
    """Kiểm tra từ toàn bộ là Kanji"""
    return all('\u4e00' <= c <= '\u9fff' for c in word)
