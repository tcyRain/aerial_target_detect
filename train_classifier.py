# -*- coding: utf-8 -*-
"""训练 / 评估航拍目标分类器（Keras 3）。"""
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
DATA_DIR = os.path.join(ROOT, "data", "classification")
OUT_DIR = os.path.join(ROOT, "output")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 96
BATCH_SIZE = 64
EPOCHS = 30
CLASS_NAMES = sorted(os.listdir(os.path.join(DATA_DIR, "train")))
N_CLASSES = len(CLASS_NAMES)


def load_split(subset):
    """用 imdecode 读图（兼容中文路径），返回 (X, Y)，X 为 float32 [0,1] RGB。"""
    d = os.path.join(DATA_DIR, subset)
    X, Y = [], []
    for i, cls in enumerate(CLASS_NAMES):
        dcls = os.path.join(d, cls)
        for f in sorted(os.listdir(dcls)):
            with open(os.path.join(dcls, f), "rb") as fh:
                img = cv2.imdecode(np.frombuffer(fh.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("imdecode failed: " + os.path.join(dcls, f))
            X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            Y.append(i)
    return np.stack(X), np.eye(N_CLASSES)[np.array(Y)].astype(np.float32)


def make_dataset(X, Y, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=42)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.10)(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv_last")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def grad_cam(model, img_batch):
    """对一批图片计算 Grad-CAM 热力图。img_batch 为 float [0,1]。"""
    conv_layer = model.get_layer("conv_last")
    grad_model = keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        idx = tf.argmax(preds, axis=1)
        one_hot = tf.one_hot(idx, depth=N_CLASSES)
        score = tf.reduce_sum(preds * one_hot, axis=1)
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam).numpy()
    for i in range(cam.shape[0]):
        c = cam[i]
        if c.max() - c.min() > 1e-8:
            cam[i] = (c - c.min()) / (c.max() - c.min())
    return cam


def main():
    print("loading train split...")
    X_train, Y_train = load_split("train")
    print("loading val split...")
    X_val, Y_val = load_split("val")
    print(f"train={X_train.shape} val={X_val.shape}")
    train_ds = make_dataset(X_train, Y_train)
    val_ds = make_dataset(X_val, Y_val)

    model = build_model()
    model.summary()
    cbs = [
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "classifier_best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max"),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "classifier_log.csv")),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)
    model.save(os.path.join(MODEL_DIR, "classifier_final.keras"))

    # 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, ylab in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        ax.plot(history.history[key], label="train")
        ax.plot(history.history["val_" + key], label="val")
        ax.set_title(ylab)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "classifier_history.png"), dpi=120)
    plt.close(fig)

    # 测试集评估
    best = keras.models.load_model(os.path.join(MODEL_DIR, "classifier_best.keras"))
    X_test, Y_test = load_split("test")
    test_ds = make_dataset(X_test, Y_test, shuffle=False)
    y_true, y_pred, probs_all = [], [], []
    sample_imgs = {}
    for imgs, labels in test_ds:
        p = best.predict(imgs, verbose=0)
        t = np.argmax(labels.numpy(), axis=1)
        pr = np.argmax(p, axis=1)
        y_true.extend(t.tolist())
        y_pred.extend(pr.tolist())
        probs_all.extend(p.tolist())
        for img, ti in zip(imgs.numpy(), t):
            if ti not in sample_imgs:
                sample_imgs[int(ti)] = img
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.mean(y_true == y_pred)
    print(f"test accuracy: {acc:.4f}")
    cm = confusion_matrix(y_true, y_pred, N_CLASSES)
    print("confusion matrix:\n", cm)
    for i, c in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i, :].sum(), 1)
        print(f"{c:10s} precision={prec:.3f} recall={rec:.3f}")

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "classifier_confusion.png"), dpi=120)
    plt.close(fig)

    # 预测样例
    order = [sample_imgs[i] for i in sorted(sample_imgs)]
    fig, axes = plt.subplots(1, len(order), figsize=(2 * len(order), 2))
    for ax, img in zip(axes, order):
        x = np.expand_dims(img.astype(np.float32) / 255.0, 0)
        p = best.predict(x, verbose=0)[0]
        label = CLASS_NAMES[int(np.argmax(p))]
        ax.imshow(img.astype(np.uint8))
        ax.set_title(f"{label} {p.max():.2f}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "classifier_samples.png"), dpi=120)
    plt.close(fig)

    # Grad-CAM
    order4 = [sample_imgs[i] for i in sorted(sample_imgs)][:4]
    x = np.stack([img.astype(np.float32) / 255.0 for img in order4])
    cam = grad_cam(best, x)
    fig, axes = plt.subplots(2, len(order4), figsize=(2.4 * len(order4), 5))
    for j, img in enumerate(order4):
        axes[0, j].imshow(img.astype(np.uint8))
        axes[0, j].set_title(CLASS_NAMES[j], fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(img.astype(np.uint8))
        axes[1, j].imshow(cam[j], cmap="jet", alpha=0.45)
        axes[1, j].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "classifier_gradcam.png"), dpi=120)
    plt.close(fig)

    print("saved:", os.path.join(MODEL_DIR, "classifier_best.keras"))
    print("outputs ->", OUT_DIR)


if __name__ == "__main__":
    main()
