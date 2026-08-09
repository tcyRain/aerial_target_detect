# 航拍目标分类与检测（Keras 实践）

面向"探测制导 / 无人机"方向的入门项目：用 Keras 3 完成

1. **目标分类**：航拍图像识别 car / building / tree / person
2. **目标检测**：YOLO 风格的轻量检测器（8x8 网格），输出类别 + 边界框

## 数据

`make_data.py` 会生成"无人机航拍视角"的合成数据集：

- `data/classification/{train,val,test}/{类别}/*.jpg`：每类 train 800 / val 100 / test 100，96x96
- `data/detection/{images,labels}/{train,val}/*`：600 / 150 张图，每张 1-3 个目标，
  标签为 YOLO 格式（`类别 cx cy w h`，归一化）
- 类别：`car` `building` `tree` `person`

真实数据（可选）：`download_real_data.py` 下载 UCMerced_LandUse 航拍分类数据集（约 317MB）。

## 运行

```bash
# 1. 生成数据
python make_data.py

# 2. 训练分类器（输出准确率、混淆矩阵、Grad-CAM 热力图）
python train_classifier.py

# 3. 训练检测器（输出 precision/recall、检测可视化）
python train_detector.py

# 4. 单独重新评估检测器
python eval_detector.py
```

在本机使用虚拟环境：

```bash
C:\Users\Lenovo\py310_env\Scripts\python.exe make_data.py
```

## 输出

- `models/classifier_best.keras`、`models/detector_best.keras`
- `output/classifier_history.png`、`classifier_confusion.png`、`classifier_samples.png`、`classifier_gradcam.png`
- `output/detector_loss.png`、`detector_samples.png`、`data_preview.png`

## 与 STM32 TinyDetector 的衔接

你已有的 `stm32_object_detection/train_tiny_detector.py` 是 32x32 灰度、2 类的
TinyDetector。本项目把输入升级为 96x96 彩色、4 类真实感目标，检测头输出结构一致
（`[conf, tx, ty, tw, th, class...]`），后续可沿用 `export_weights_to_c.py` 的思路
导出到嵌入式平台。

## 下一步建议

1. 下载 UCMerced 真实航拍数据，用迁移学习（MobileNetV2 微调）替换合成分类器
2. 用无人机/手机实际拍摄 + labelImg 标注，制作自己的检测数据集
3. 把检测结果接到 PID 视觉伺服，做"发现目标 -> 跟踪 -> 降落"闭环
