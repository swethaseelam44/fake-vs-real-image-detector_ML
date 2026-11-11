# Fake vs Real Image Classifier

Binary image classifier to detect AI-generated / fake images vs real images using transfer learning.

## Overview

This project builds a deep learning model that looks at an image and predicts whether it is Real or Fake (AI-generated). It uses transfer learning with ResNet50 to achieve high accuracy even with a limited dataset.

This was completed as part of my Master of Data Science at Flinders University.

## Tech Stack

- Python
- TensorFlow / Keras
- ResNet50 (pre-trained on ImageNet)
- ImageDataGenerator
- NumPy
- split-folders

## How It Works

1. Transfer Learning: Used ResNet50 pre-trained on ImageNet as an encoder / feature extractor.
2. Fine-tuning: Froze all layers except the last 20 layers, and un-froze the last 20 layers so the model can learn the difference between fake vs real.
3. Data Split: Split the dataset into train / val / test sets using split-folders.
4. Data Augmentation: Applied rotation, width/height shift, shear, zoom, horizontal flip to prevent overfitting.
5. Feature Encoding: Encoded images with ResNet50 and saved the features as .npy files for efficient training and to reduce memory usage.
6. Classification: Binary classification - Fake (0) vs Real (1).

## Dataset Structure

data/
train/ Real/ Fake/
val/ Real/ Fake/
test/ Real/ Fake/

Note: Dataset is not included in this repo due to size.

## How to Run

pip install -r requirements.txt
jupyter notebook fake_real_detector.ipynb

## Model Details

- Input Size: 224 x 224 x 3
- Batch Size: 32 (training) / 16 (encoding)
- Base Model: ResNet50, include_top=False, weights=imagenet
- Additional Layers: GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
- Class Mode: Binary

## Results

- Train set: 8739
- Validation set: 1092
- Test set: 1094
- Test Accuracy: 70%

## Author

Swetha Seelam
Master of Data Science, Flinders University
LinkedIn: linkedin.com/in/swetha-seelam-588144350
GitHub: github.com/swethaseelam44
