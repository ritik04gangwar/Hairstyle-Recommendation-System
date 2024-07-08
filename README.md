
# Deep Learning-based Hairstyle Recommendation System

This repository contains a project for a hairstyle recommendation system, which uses a deep learning neural network built from scratch. The system classifies the face shape of a given input image into one of five categories: heart, oblong, oval, round, or square, and then recommends 10 hairstyles based on the face shape.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Introduction

Choosing a hairstyle that suits one's face shape can be challenging. This project aims to simplify this process by leveraging deep learning to automatically classify face shapes and provide hairstyle recommendations accordingly. The system can classify faces into five shapes: heart, oblong, oval, round, and square, and recommend 10 suitable hairstyles for each shape.

## Dataset

The dataset used for this project includes images of faces labeled with their respective shapes (heart, oblong, oval, round, or square). The dataset is divided into training, validation, and test sets to ensure robust model performance.

## Model Architecture

The deep learning model is a convolutional neural network (CNN) designed and built from scratch to handle image classification tasks. The architecture includes:
- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification

## Training

The model is trained using the labeled dataset. The training process includes data augmentation to increase variability, loss calculation, and optimization using backpropagation. The training is monitored through validation accuracy and loss.

## Evaluation

The model's performance is evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and ROC curves are also utilized to visualize performance.

## Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
