# ==== BUILD EfficientNet-B3 ====
# effnet_version_of_code2.py
import os
import json
import threading
from collections import Counter, OrderedDict
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ====== HUGGINGFACE CONFIG ======# ==== CONFIG ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
TOPK = 5
REPO_ID = "TuanVu219/Vit_Checkpoint_New"
EFFNET_FILENAME = "best_effnet_ema.pth"

effnet_ckpt_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=EFFNET_FILENAME
)

# ====== CLASS LABELS ======
with open("core/class_names.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

idx2label = {i: l for i, l in enumerate(labels)}
num_classes = len(labels)

# ====== DEVICE ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==========================================================
#     🎯 build_efficientnet_b3 — tên hàm bạn yêu cầu
# ==========================================================
def build_efficientnet_b3(num_classes, pretrained=True, device=None):
    """
    Xây EfficientNet-B3 với classifier được thay bằng đúng số class.
    """
    if pretrained:
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
    else:
        model = models.efficientnet_b3(weights=None)

    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

    if device:
        model.to(device)

    return model


# ==========================================================
#     🎯 load_checkpoint_to_model — giữ nguyên tên cũ
# ==========================================================
def load_checkpoint_to_model(model, ckpt_path, use_ema_if_available=True, map_location=None):
    print("Loading EfficientNet checkpoint:", ckpt_path)

    map_location = map_location or device
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)

    # Chọn state_dict
    if isinstance(ckpt, dict):
        if use_ema_if_available and "ema" in ckpt:
            sd = ckpt["ema"]
            print("🔁 Loading EMA weights")
        elif "model" in ckpt:
            sd = ckpt["model"]
            print("🔁 Loading model weights")
        else:
            sd = ckpt
            print("🔁 Loading raw state_dict")
    else:
        sd = ckpt

    # Loại bỏ phần "module."
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_sd[k.replace("module.", "", 1)] = v
        else:
            new_sd[k] = v

    # Load state_dict
    try:
        model.load_state_dict(new_sd, strict=False)
        print("✅ state_dict loaded (strict=False)")
    except Exception as e:
        print("⚠ load_state_dict error:", e)
        model_dict = model.state_dict()

        partial = {k: v for k, v in new_sd.items()
                   if k in model_dict and v.size() == model_dict[k].size()}

        model_dict.update(partial)
        model.load_state_dict(model_dict)
        print("✅ Partial weights loaded")

    model.to(device)
    model.eval()
    return model


# ==========================================================
#     🎯 get_effnet_model — tên cuối cùng bạn sẽ dùng
# ==========================================================
_model = None
_model_lock = threading.Lock()

def get_effnet_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("🔹 Loading EfficientNet model (only once)...")

                model = build_efficientnet_b3(
                    num_classes=num_classes,
                    pretrained=True,
                    device=device
                )

                model = load_checkpoint_to_model(
                    model,
                    effnet_ckpt_path,
                    use_ema_if_available=True
                )

                # 🔥 WARMUP — CỰC QUAN TRỌNG
                dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
                with torch.no_grad():
                    model(dummy)

                _model = model

    return _model



# ==== TRANSFORM ====
preprocess_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_canvas_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return preprocess_transform(img).unsqueeze(0).to(DEVICE)

# ==== RECOGNITION ====
def recognize_char(img, k=TOPK):
    model = get_effnet_model()
    x = preprocess_canvas_image(img)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, k)
    results = [(idx2label[top_idxs[i].item()], float(top_probs[i].item())) for i in range(k)]
    return results


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

def recognize_and_merge_boxes_exhaustive(
    gray,
    boxes,
    max_span=None,
    k=TOPK
):
    n = len(boxes)
    merged_boxes = []
    i = 0
    cache = {}
    H, W = gray.shape[:2]

    while i < n:
        best_prob = -1.0
        best_box = None
        best_j = i
        best_preds = None

        # init merged box
        nx, ny, nx2, ny2 = (
            boxes[i][0],
            boxes[i][1],
            boxes[i][0] + boxes[i][2],
            boxes[i][1] + boxes[i][3]
        )

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
                crop = Image.fromarray(
                    gray[y1:y2c, x1:x2c]
                ).convert("RGB")
                preds = recognize_char(crop, k=k)
                cache[key] = preds

            if not preds:
                continue

            top_label, top_p = preds[0]
            if top_p > best_prob:
                best_prob = top_p
                best_box = [x1, y1, x2c - x1, y2c - y1]
                best_j = j
                best_preds = preds

        # fallback cực hiếm: model không trả gì
        if best_box is None:
            crop = Image.fromarray(
                gray[
                    boxes[i][1]:boxes[i][1] + boxes[i][3],
                    boxes[i][0]:boxes[i][0] + boxes[i][2]
                ]
            ).convert("RGB")
            best_preds = recognize_char(crop, k=k)
            best_box = boxes[i]
            best_j = i

        merged_boxes.append((best_box, best_preds))
        i = best_j + 1

    return merged_boxes


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
    merged = recognize_and_merge_boxes_exhaustive(gray, boxes, k=k)
    return merged

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
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness=15, lineType=cv2.LINE_AA)
    pil_img = Image.fromarray(img)
    folder_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(folder_path, filename)
    pil_img.save(save_path)
    print(f"✅ Image saved to {save_path}")
    return pil_img

# =========================================================
# HỖ TRỢ KIỂM TRA KANJI
# =========================================================
def is_kanji(word):
    return all('\u4e00' <= c <= '\u9fff' for c in word)
