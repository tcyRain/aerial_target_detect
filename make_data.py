# -*- coding: utf-8 -*-
"""
生成"无人机航拍视角"合成数据集：
  1) 分类: data/classification/{train,val,test}/{car,building,tree,person}/*.jpg (96x96)
  2) 检测: data/detection/{images,labels}/{train,val}/*.jpg + *.txt (YOLO 格式)

运行:
  python make_data.py
"""
import os
import numpy as np
import cv2

SEED = 42
IMG_SIZE = 96
CLASSES = ["car", "building", "tree", "person"]

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# 分类数据集规模（每类）
CLS_TRAIN_PER_CLASS = 800
CLS_VAL_PER_CLASS = 100
CLS_TEST_PER_CLASS = 100
# 检测数据集规模
DET_TRAIN = 600
DET_VAL = 150


def make_background(rng):
    """随机地形背景：沥青 / 草地 / 土地 / 屋顶。"""
    base = [(105, 108, 112), (72, 118, 58), (120, 96, 62), (142, 104, 78)][
        int(rng.integers(0, 4))
    ]
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    img[:] = base
    for _ in range(int(rng.integers(20, 45))):
        c = tuple(int(np.clip(v + rng.integers(-25, 26), 0, 255)) for v in base)
        r = int(rng.integers(2, 8))
        cv2.circle(img, (int(rng.integers(0, IMG_SIZE)), int(rng.integers(0, IMG_SIZE))), r, c, -1)
    noise_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
    noise = noise_rng.normal(0, 6, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
    d = np.sqrt((xx - IMG_SIZE / 2.0) ** 2 + (yy - IMG_SIZE / 2.0) ** 2) / (IMG_SIZE / 2.0)
    gain = 1.0 - 0.28 * np.clip(d, 0, 1.15)
    img = np.clip(img.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)
    return img


def render_object(cls, w, h, rng):
    """在正方形透明画布上绘制单个目标，返回 (patch, mask)。"""
    pad = 6
    S = int(np.hypot(w, h)) + 2 * pad
    patch = np.zeros((S, S, 3), np.uint8)
    mask = np.zeros((S, S), np.uint8)
    cx, cy = S // 2, S // 2

    if cls == "car":
        bw, bh = int(w), max(2, int(h * 0.62))
        body = tuple(int(rng.integers(80, 255)) for _ in range(3))
        cv2.rectangle(patch, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), body, -1)
        cv2.rectangle(mask, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), 255, -1)
        win = tuple(int(v * 0.5) for v in body)
        wy1 = cy - bh // 2 + 2
        wy2 = wy1 + 3 + max(2, bh // 4)
        cv2.rectangle(patch, (cx - bw // 2 + 2, wy1), (cx + bw // 2 - 2, wy2), win, -1)
        cv2.rectangle(mask, (cx - bw // 2 + 2, wy1), (cx + bw // 2 - 2, wy2), 255, -1)
        light = tuple(min(255, int(v * 1.2)) for v in body)
        cv2.line(patch, (cx - bw // 2 + 3, cy - bh // 4), (cx + bw // 2 - 3, cy - bh // 4), light, 1)

    elif cls == "building":
        bw, bh = int(w), int(h)
        roof = tuple(int(rng.integers(90, 210)) for _ in range(3))
        cv2.rectangle(patch, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), roof, -1)
        cv2.rectangle(mask, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), 255, -1)
        dark = tuple(int(v * 0.7) for v in roof)
        cv2.rectangle(patch, (cx - bw // 2 + 1, cy - bh // 2 + 1),
                      (cx + bw // 2 - 1, cy + bh // 2 - 1), dark, 1)
        for _ in range(int(rng.integers(1, 4))):
            ex = cx + int(rng.integers(-bw // 4, bw // 4 + 1))
            ey = cy + int(rng.integers(-bh // 4, bh // 4 + 1))
            es = int(rng.integers(2, max(3, min(bw, bh) // 6)))
            hi = tuple(min(255, int(v * 1.15)) for v in roof)
            cv2.rectangle(patch, (ex - es, ey - es), (ex + es, ey + es), hi, -1)
            cv2.rectangle(mask, (ex - es, ey - es), (ex + es, ey + es), 255, -1)

    elif cls == "tree":
        r = max(3, int(min(w, h) / 2))
        green = (int(rng.integers(35, 75)), int(rng.integers(95, 160)), int(rng.integers(30, 70)))
        cv2.circle(patch, (cx, cy), r, green, -1)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        for _ in range(int(rng.integers(4, 9))):
            a = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(0.3, 0.8) * r
            px = int(cx + rr * np.cos(a))
            py = int(cy + rr * np.sin(a))
            pr = int(rng.uniform(0.2, 0.5) * r)
            cv2.circle(patch, (px, py), pr, green, -1)
            cv2.circle(mask, (px, py), pr, 255, -1)
        hi = tuple(min(255, int(v * 1.25)) for v in green)
        cv2.circle(patch, (cx - r // 3, cy - r // 3), max(2, r // 4), hi, -1)

    else:  # person，俯视视角：肩部椭圆 + 头部圆
        bw, bh = int(w), int(h)
        body = tuple(int(rng.integers(25, 70)) for _ in range(3))
        cv2.ellipse(patch, (cx, cy + 2), (bw // 2, int(bh * 0.35)), 0, 0, 360, body, -1)
        cv2.ellipse(mask, (cx, cy + 2), (bw // 2, int(bh * 0.35)), 0, 0, 360, 255, -1)
        hr = max(2, int(bw * 0.28))
        cv2.circle(patch, (cx, cy - int(bh * 0.18)), hr, body, -1)
        cv2.circle(mask, (cx, cy - int(bh * 0.18)), hr, 255, -1)

    return patch, mask


def paste(img, patch, mask, cx_px, cy_px):
    """带旋转后的 patch+mask 合成到背景图。"""
    S = patch.shape[0]
    x0 = int(cx_px - S / 2)
    y0 = int(cy_px - S / 2)
    x1, y1 = x0 + S, y0 + S
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, IMG_SIZE), min(y1, IMG_SIZE)
    if x1c <= x0c or y1c <= y0c:
        return
    patch_c = patch[y0c - y0: y1c - y0, x0c - x0: x1c - x0]
    mask_c = mask[y0c - y0: y1c - y0, x0c - x0: x1c - x0]
    roi = img[y0c:y1c, x0c:x1c]
    roi[mask_c > 0] = patch_c[mask_c > 0]
    img[y0c:y1c, x0c:x1c] = roi


def place_object(img, cls, cx_px, cy_px, w_px, h_px, angle, rng):
    """先画影子，再画旋转后的目标。"""
    cv2.ellipse(img, (int(cx_px + 2), int(cy_px + 3)),
                (max(2, int(w_px * 0.55)), max(2, int(h_px * 0.5))),
                0, 0, 360, (25, 25, 25), -1)
    patch, mask = render_object(cls, w_px, h_px, rng)
    M = cv2.getRotationMatrix2D((patch.shape[1] / 2.0, patch.shape[0] / 2.0), float(angle), 1.0)
    patch_r = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]))
    mask_r = cv2.warpAffine(mask, M, (patch.shape[1], patch.shape[0]))
    paste(img, patch_r, mask_r, cx_px, cy_px)


def finish(img, rng):
    """轻度模糊 + 亮度对比度抖动。"""
    if rng.random() < 0.35:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    alpha = rng.uniform(0.85, 1.15)
    beta = int(rng.integers(-12, 13))
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def save_image(path, img):
    """用 imencode + 二进制写入，避免 cv2.imwrite 在中文路径下静默失败。"""
    ext = os.path.splitext(path)[1]
    ret, buf = cv2.imencode(ext, img)
    if not ret:
        raise RuntimeError("imencode failed: " + path)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def load_image(path):
    """用 imdecode 读取，避免 cv2.imread 在中文路径下返回 None。"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("imdecode failed: " + path)
    return img


def gen_classification(rng):
    sizes = {"car": (28, 42), "building": (34, 52), "tree": (24, 38), "person": (16, 24)}
    for split, n_per in [
        ("train", CLS_TRAIN_PER_CLASS),
        ("val", CLS_VAL_PER_CLASS),
        ("test", CLS_TEST_PER_CLASS),
    ]:
        for cls in CLASSES:
            d = os.path.join(DATA_DIR, "classification", split, cls)
            os.makedirs(d, exist_ok=True)
            lo, hi = sizes[cls]
            for i in range(n_per):
                img = make_background(rng)
                w_px = int(rng.integers(lo, hi + 1))
                if cls == "tree":
                    h_px = w_px
                elif cls == "car":
                    h_px = int(w_px * 0.62)
                elif cls == "building":
                    h_px = int(w_px * 0.85)
                else:
                    h_px = int(w_px * 0.9)
                margin = int(np.hypot(w_px, h_px) / 2) + 8
                half = IMG_SIZE / 2 - margin
                cx_px = IMG_SIZE / 2 + rng.uniform(-half, half)
                cy_px = IMG_SIZE / 2 + rng.uniform(-half, half)
                place_object(img, cls, cx_px, cy_px, w_px, h_px, rng.uniform(0, 360), rng)
                img = finish(img, rng)
                save_image(os.path.join(d, f"{split}_{cls}_{i:04d}.jpg"), img)
    print("classification dataset done")


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def gen_detection(rng):
    norm_sizes = {"car": (0.13, 0.24), "building": (0.18, 0.32),
                  "tree": (0.11, 0.20), "person": (0.06, 0.11)}
    wh_ratio = {"car": 0.62, "building": 0.85, "tree": 1.0, "person": 0.9}
    for split, n in [("train", DET_TRAIN), ("val", DET_VAL)]:
        img_dir = os.path.join(DATA_DIR, "detection", "images", split)
        lab_dir = os.path.join(DATA_DIR, "detection", "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        for i in range(n):
            img = make_background(rng)
            boxes = []
            for _ in range(int(rng.integers(1, 4))):
                for _attempt in range(10):
                    cls = CLASSES[int(rng.integers(0, len(CLASSES)))]
                    lo, hi = norm_sizes[cls]
                    w = rng.uniform(lo, hi)
                    h = w * wh_ratio[cls]
                    cx = rng.uniform(w / 2 + 0.03, 1 - w / 2 - 0.03)
                    cy = rng.uniform(h / 2 + 0.03, 1 - h / 2 - 0.03)
                    if not any(bbox_iou((cx, cy, w, h), b) > 0.40 for b in boxes):
                        break
                boxes.append((CLASSES.index(cls), cx, cy, w, h))
                place_object(img, cls, cx * IMG_SIZE, cy * IMG_SIZE,
                             w * IMG_SIZE, h * IMG_SIZE, rng.uniform(0, 360), rng)
            img = finish(img, rng)
            name = f"{split}_{i:04d}"
            save_image(os.path.join(img_dir, name + ".jpg"), img)
            with open(os.path.join(lab_dir, name + ".txt"), "w") as f:
                for cid, cx, cy, w, h in boxes:
                    f.write(f"{cid} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")
    with open(os.path.join(DATA_DIR, "detection", "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES))
    print("detection dataset done")


def preview():
    rows = []
    for cls in CLASSES:
        d = os.path.join(DATA_DIR, "classification", "train", cls)
        files = sorted(os.listdir(d))[:4]
        imgs = [load_image(os.path.join(d, f)) for f in files]
        imgs[0] = cv2.putText(imgs[0], cls, (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                              0.4, (0, 255, 255), 1)
        rows.append(np.hstack(imgs))
    cls_grid = np.vstack(rows)

    det_imgs = []
    d = os.path.join(DATA_DIR, "detection", "images", "val")
    for f in sorted(os.listdir(d))[:4]:
        img = load_image(os.path.join(d, f))
        with open(os.path.join(DATA_DIR, "detection", "labels", "val",
                               os.path.splitext(f)[0] + ".txt")) as fp:
            for line in fp:
                p = line.split()
                cid, cx, cy, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x1 = int((cx - w / 2) * IMG_SIZE)
                y1 = int((cy - h / 2) * IMG_SIZE)
                x2 = int((cx + w / 2) * IMG_SIZE)
                y2 = int((cy + h / 2) * IMG_SIZE)
                color = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 255, 255)][cid]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                cv2.putText(img, CLASSES[cid], (x1, max(0, y1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        det_imgs.append(img)
    det_grid = np.hstack(det_imgs)
    grid = np.vstack([cls_grid, det_grid])
    out = os.path.join(OUT_DIR, "data_preview.png")
    save_image(out, grid)
    print("preview saved:", out)


def main():
    rng = np.random.default_rng(SEED)
    gen_classification(rng)
    gen_detection(rng)
    preview()
    print("ALL DONE")


if __name__ == "__main__":
    main()
