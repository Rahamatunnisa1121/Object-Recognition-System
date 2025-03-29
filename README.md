# COCO Object Recognition System

## 📌 Overview
This project implements an **object recognition system** using the **COCO dataset**, a large-scale dataset for object detection, segmentation, and captioning tasks. The system utilizes deep learning models to detect and classify objects in images.
## Usage
To run the segmentation model, use the following command:
### `streamlit run app.py`

## Segmentation
This project extends object recognition by incorporating instance segmentation, allowing precise pixel-wise segmentation of objects in images. Using deep learning models trained on the COCO dataset, the system not only detects objects but also generates segmentation masks to differentiate between overlapping objects and background areas.

## Features
- **Instance Segmentation** : Generates detailed object masks instead of just bounding boxes.
- **COCO Dataset Integration** : Utilizes pre-trained models for segmentation tasks.
- **Deep Learning Models** : Leverages state-of-the-art architectures like Mask R-CNN for accurate segmentation.
- **Visualization Tools** : Displays segmented objects with color-coded masks for better interpretability.

## Usage
To run the segmentation model, use the following command:
### `streamlit run segmentation_app.py`


## 📂 Dataset Structure
The dataset follows the **COCO format** and consists of:
- **Annotations Folder** (`.json` files): Contains metadata like bounding boxes, segmentation masks, and object categories.
- **Image Folder**: Stores images corresponding to the annotations.
- **Categories**: Different object classes labeled with unique category IDs.
- **Annotations**: Each image can contain multiple annotated objects, leading to **more annotations than images**.

---

## ⚙️ Installation & Requirements  

### 🛠 Required Libraries:
To run this project, you need the following dependencies:

pip install tensorflow torch torchvision opencv-python numpy matplotlib seaborn pycocotools scikit-learn pillow ultralytics

![Yolo Inference](https://raw.githubusercontent.com/Rahamatunnisa1121/Object-Recognition-System/main/yolo_test.png)
