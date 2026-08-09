# -*- coding: utf-8 -*-
"""加载训练好的检测器，重新对验证集做评估与可视化。"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

import train_detector as td


def main():
    model = keras.models.load_model(
        os.path.join(td.MODEL_DIR, "detector_best.keras"),
        custom_objects={"yolo_loss": td.yolo_loss},
    )
    X_val, Y_val = td.load_split("val")
    best_thresh, best_f1 = 0.35, 0.0
    for th in [0.25, 0.35, 0.45, 0.55, 0.65]:
        precision, recall = td.eval_detector(model, X_val, Y_val, conf_thresh=th)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"  threshold={th}: f1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, th
    print(f"best threshold={best_thresh} f1={best_f1:.3f}")

    idx = np.linspace(0, len(X_val) - 1, 9).astype(int)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, i in zip(axes.flat, idx):
        gt_boxes = []
        for row in range(td.GRID):
            for col in range(td.GRID):
                if Y_val[i, row, col, 0] > 0.5:
                    cls = int(np.argmax(Y_val[i, row, col, 5:]))
                    tx, ty, tw, th = Y_val[i, row, col, 1:5]
                    cx = (col + tx) / td.GRID
                    cy = (row + ty) / td.GRID
                    w = np.exp(tw) / td.GRID
                    h = np.exp(th) / td.GRID
                    gt_boxes.append((cls, cx, cy, w, h))
        img = td.draw_boxes(X_val[i], gt_boxes, gt=True)
        pred_boxes = td.decode(model.predict(X_val[i:i + 1], verbose=0)[0], conf_thresh=best_thresh)
        img = td.draw_boxes(img, pred_boxes, gt=False)
        ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(td.OUT_DIR, "detector_samples.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
