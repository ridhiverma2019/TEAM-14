# TEAM-14

# Image Classification on Tiny-ImageNet: Comparing Traditional and Deep Learning Features

This project explores various image feature extraction techniques and analyzes their impact on classification performance across different machine learning models. The goal is to compare the effectiveness of traditional, handcrafted features against features learned automatically by deep convolutional neural networks (CNNs).

## Objective

The primary objective is to investigate how different feature extraction methods influence the performance of various classifiers on a multi-class image classification task. By comparing traditional methods like LBP, HOG, and Canny Edge Detection with deep learning-based approaches using VGG16 and ResNet50, we aim to understand the strengths and weaknesses of each approach.

## Dataset

The project uses a curated subset of the Tiny-ImageNet dataset, including 12 diverse classes of images such as "African elephant," "sports car," "monarch," and "koala." The data was collected using an automated downloader script that validated ImageNet URLs to ensure a robust and usable dataset. The dataset is split into training and testing sets with a stratified 80/20 ratio.

## Feature Extraction Methods

The project evaluates two main categories of feature extraction techniques: traditional and deep learning-based methods.

### Traditional Methods

- **Local Binary Patterns (LBP)**  
  A simple yet effective texture descriptor that captures local micro-patterns by thresholding neighbors against a central pixel. Multi-scale LBP was used with radii of 1, 2, and 3. PCA was then applied to reduce the dimensionality of the resulting feature vectors.

- **Histogram of Oriented Gradients (HOG)**  
  A descriptor that captures local edge and shape structure by analyzing gradient directions in small image regions. HOG is robust to small geometric and photometric changes.

- **Canny Edge Detection**  
  A multi-stage algorithm that identifies a wide range of edges in images, providing a feature set based on object outlines.

### Deep Learning Methods

Pretrained CNN models were used as feature extractors via transfer learning. The models' fully connected layers were removed to extract rich, hierarchical feature maps from the final convolutional layer.

- **VGG16**  
  Extracts a flattened feature vector of 25,088 dimensions from a 7x7x512 feature map. VGG16 features capture fine-grained textures and edges.

- **ResNet50**  
  A deeper network that extracts a feature vector of 100,352 dimensions from a 7x7x2048 feature map. ResNet50 features encode more abstract, high-level semantic information.

## Classification and Results

The extracted features were fed into four different machine learning classifiers:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Random Forest  

### Key Findings

- **Deep Learning Features:** Consistently and significantly outperform traditional features across all classifiers.  

- **VGG16 Features:** Achieved the highest performance metrics, with Logistic Regression reaching an accuracy of 81.35% and an F1 score of 0.80. Random Forest also showed exceptional performance.  

- **ResNet50 Features:** While powerful, they performed slightly worse than VGG16 features, suggesting they may be too abstract for these classifiers or this specific dataset without fine-tuning.  

- **Traditional Features:** All traditional methods delivered moderate to poor results. Canny Edge Detection features performed the worst, with low accuracy and high log loss, as they lacked sufficient information for classification.

This project confirms the superiority of deep learning for feature extraction in image classification, as the learned hierarchical representations are far more expressive and robust than handcrafted features.
