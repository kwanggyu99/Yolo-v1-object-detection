# YOLO v1 with ResNet Backbone

This repository contains an implementation of the YOLO v1 object detection model, modified to use ResNet as the backbone network. The modification aims to enhance the model's performance in detecting objects in images.

## Introduction

YOLO (You Only Look Once) is a popular real-time object detection model. In this project, we've implemented YOLO v1 with a ResNet backbone instead of the original YOLO backbone to improve the model's accuracy and robustness. The ResNet architecture is known for its deep learning capabilities, which helps in capturing more complex features from images.

## Model Architecture

### YOLO v1

YOLO v1 divides the input image into a grid and predicts bounding boxes and class probabilities directly from the full images in one evaluation. This approach makes YOLO extremely fast compared to other object detection models.

### ResNet Backbone

ResNet (Residual Networks) is a deep convolutional neural network architecture that mitigates the vanishing gradient problem, allowing for very deep networks. By using ResNet as the backbone for YOLO v1, the model benefits from enhanced feature extraction, leading to improved detection performance.
