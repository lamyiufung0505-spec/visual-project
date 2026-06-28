

## 项目简介

本项目为《计算机视觉基础》课程实验。

实验包含两个部分：

* 基于CLIP模型的图像检索(Image Retrieval)
* 基于EasyOCR的图像文字识别(OCR)

开发环境：

* Python 3.x
* PyTorch
* OpenCLIP
* EasyOCR

---

## 项目结构

```
.
├── clip_retrieval.py    # 图像检索程序
├── ocr_test.py          # OCR文字识别程序
├── README.md
└── dataset
    ├── image_retrieval
    └── object_detection
```

---

## 安装依赖

```bash
pip install torch torchvision
pip install open_clip_torch
pip install easyocr
pip install pillow numpy tqdm matplotlib
```

---

## 运行方法

### 1. 图像检索

```bash
python clip_retrieval.py
```

程序将读取Base图像库，为Query图片检索Top-5最相似结果。

---

### 2. OCR文字识别

```bash
python ocr_test.py
```

程序将读取数据集中的图片，并输出识别出的文字信息。

---

## 实验结果

实验成功完成了：

* CLIP图像检索
* OCR文字识别

图像检索能够返回查询图片对应的Top-5相似图片；OCR能够识别图片中的文字信息。

---

## 作者

课程实验项目，仅用于学习交流。
