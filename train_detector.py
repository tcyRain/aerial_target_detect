# -*- coding: utf-8 -*-
"""训练 YOLO 风格的轻量检测器（Keras 3，8x8 网格，4 类）。

输出张量 (8, 8, 9): [conf, tx, ty, tw, th, cls0, cls1, cls2, cls3]
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "detection")
OUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 96
GRID = 8
NUM_CLASSES = 4
OUT_CH = 5 + NUM_CLASSES
COORD = 5.0
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = 32
EPOCHS = 150
CLASS_NAMES = ["car", "building", "tree", "person"]
COLORS = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 255, 255)]


def encode(boxes):
    """把归一化框编码成网格标签。"""
    label = np.zeros((GRID, GRID, OUT_CH), np.float32)
    for cls, cx, cy, w, h in boxes:
        col = min(int(cx * GRID), GRID - 1)
        row = min(int(cy * GRID), GRID - 1)
        tx = cx * GRID - col
        ty = cy * GRID - row
        tw = np.log(max(w * GRID, 1e-6))
        th = np.log(max(h * GRID, 1e-6))
        label[row, col, 0] = 1.0
        label[row, col, 1] = tx
        label[row, col, 2] = ty
        label[row, col, 3] = tw
        label[row, col, 4] = th
        label[row, col, 5 + cls] = 1.0
    return label


def load_split(split):
    img_dir = os.path.join(DATA_DIR, "images", split)
    lab_dir = os.path.join(DATA_DIR, "labels", split)
    names = sorted(os.listdir(img_dir))
    X, Y = [], []
    for n in names:
        with open(os.path.join(img_dir, n), "rb") as f:
            data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("imdecode failed: " + os.path.join(img_dir, n))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.astype(np.float32) / 255.0)
        boxes = []
        with open(os.path.join(lab_dir, os.path.splitext(n)[0] + ".txt")) as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        Y.append(encode(boxes))
    return np.stack(X), np.stack(Y)


def yolo_loss(y_true, y_pred):
    obj = y_true[..., 0:1]
    xy = tf.reduce_sum(obj * tf.square(y_true[..., 1:3] - y_pred[..., 1:3]))
    wh = tf.reduce_sum(obj * tf.square(y_true[..., 3:5] - y_pred[..., 3:5]))
    noobj = 1.0 - obj
    conf_bce = -(y_true[..., 0:1] * tf.math.log_sigmoid(y_pred[..., 0:1]) +
                 (1.0 - y_true[..., 0:1]) * tf.math.log_sigmoid(-y_pred[..., 0:1]))
    conf = tf.reduce_sum(obj * conf_bce) + LAMBDA_NOOBJ * tf.reduce_sum(noobj * conf_bce)
    # 手写 BCE（类别输出为 logits），避免 Keras 3 loss 函数的形状歧义
    bce = -(y_true[..., 5:] * tf.math.log_sigmoid(y_pred[..., 5:]) +
            (1.0 - y_true[..., 5:]) * tf.math.log_sigmoid(-y_pred[..., 5:]))
    cls = tf.reduce_sum(obj * tf.reduce_sum(bce, axis=-1, keepdims=True))
    n_obj = tf.maximum(tf.reduce_sum(obj), 1.0)
    total = COORD * (xy + wh) + conf + cls
    return total / tf.cast(tf.shape(y_true)[0], tf.float32)


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(OUT_CH, 1, padding="same")(x)
    x = layers.Resizing(GRID, GRID)(x)
    model = keras.Model(inputs, x)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=yolo_loss)
    return model


def decode(pred, conf_thresh=0.35):
    """把 (G,G,C) 输出解码成检测框列表。"""
    pred = np.squeeze(pred)
    boxes = []
    for row in range(GRID):
        for col in range(GRID):
            conf = 1.0 / (1.0 + np.exp(-pred[row, col, 0]))
            if conf < conf_thresh:
                continue
            tx, ty, tw, th = pred[row, col, 1:5]
            cx = (col + tx) / GRID
            cy = (row + ty) / GRID
            w = np.exp(tw) / GRID
            h = np.exp(th) / GRID
            cls = int(np.argmax(pred[row, col, 5:]))
            boxes.append((cls, cx, cy, w, h, float(conf)))
    return nms(boxes)


def nms(boxes, iou_thresh=0.35):
    """按置信度降序贪心抑制同一目标上的重复框。"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    keep = []
    for b in boxes:
        if any(b[0] == k[0] and bbox_iou(b, k) > iou_thresh for k in keep):
            continue
        keep.append(b)
    return keep


def bbox_iou(a, b):
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[3] * a[4] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def draw_boxes(img_rgb, boxes, gt=False):
    """画框。gt=True 时画绿色框，否则画彩色框。"""
    img = cv2.cvtColor((np.clip(img_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    for b in boxes:
        cls, cx, cy, w, h = b[0], b[1], b[2], b[3], b[4]
        x1 = int((cx - w / 2) * IMG_SIZE)
        y1 = int((cy - h / 2) * IMG_SIZE)
        x2 = int((cx + w / 2) * IMG_SIZE)
        y2 = int((cy + h / 2) * IMG_SIZE)
        if gt:
            color = (0, 255, 0)
            label = CLASS_NAMES[cls]
        else:
            color = COLORS[cls]
            label = f"{CLASS_NAMES[cls]} {b[5]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def eval_detector(model, X, Y, conf_thresh=0.35):
    preds = model.predict(X, verbose=0)
    tp = fp = fn = 0
    for i in range(len(X)):
        gt_boxes = []
        for row in range(GRID):
            for col in range(GRID):
                if Y[i, row, col, 0] > 0.5:
                    cls = int(np.argmax(Y[i, row, col, 5:]))
                    tx, ty, tw, th = Y[i, row, col, 1:5]
                    cx = (col + tx) / GRID
                    cy = (row + ty) / GRID
                    w = np.exp(tw) / GRID
                    h = np.exp(th) / GRID
                    gt_boxes.append((cls, cx, cy, w, h))
        pred_boxes = decode(preds[i], conf_thresh=conf_thresh)
        used = set()
        for pb in pred_boxes:
            best_iou, best_gt = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in used:
                    continue
                iou = bbox_iou(pb, gb)
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            if best_iou >= 0.3 and best_gt >= 0 and pb[0] == gt_boxes[best_gt][0]:
                tp += 1
                used.add(best_gt)
            else:
                fp += 1
        fn += len(gt_boxes) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"[val] precision={precision:.3f} recall={recall:.3f} "
          f"(tp={tp}, fp={fp}, fn={fn})")
    return precision, recall


def main():
    print("loading train split...")
    X_train, Y_train = load_split("train")
    print("loading val split...")
    X_val, Y_val = load_split("val")
    print(f"train={X_train.shape} val={X_val.shape}")
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)) \
        .shuffle(len(X_train), seed=42).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val)).batch(BATCH_SIZE)

    model = build_model()
    model.summary()
    cbs = [
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "detector_best.keras"),
            monitor="val_loss", save_best_only=True, mode="min"),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "detector_log.csv")),
        callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=cbs,
        verbose=1,
    )
    model.save(os.path.join(MODEL_DIR, "detector_final.keras"))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title("Detector loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "detector_loss.png"), dpi=120)
    plt.close(fig)

    best = keras.models.load_model(
        os.path.join(MODEL_DIR, "detector_best.keras"),
        custom_objects={"yolo_loss": yolo_loss},
    )
    eval_detector(best, X_val, Y_val)

    # 可视化: 取 9 张验证图, 绿=真值, 彩色=预测
    idx = np.linspace(0, len(X_val) - 1, 9).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, i in zip(axes.flat, idx):
        gt_boxes = []
        for row in range(GRID):
            for col in range(GRID):
                if Y_val[i, row, col, 0] > 0.5:
                    cls = int(np.argmax(Y_val[i, row, col, 5:]))
                    tx, ty, tw, th = Y_val[i, row, col, 1:5]
                    cx = (col + tx) / GRID
                    cy = (row + ty) / GRID
                    w = np.exp(tw) / GRID
                    h = np.exp(th) / GRID
                    gt_boxes.append((cls, cx, cy, w, h))
        img = draw_boxes(X_val[i], gt_boxes, gt=True)
        pred_boxes = decode(best.predict(X_val[i:i + 1], verbose=0)[0])
        img = draw_boxes(img, pred_boxes, gt=False)
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "detector_samples.png"), dpi=120)
    plt.close(fig)
    print("saved:", os.path.join(MODEL_DIR, "detector_best.keras"))


if __name__ == "__main__":
    main()
