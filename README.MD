# Unsupervised Layer-wise Pretraining in Deep Neural Networks

This implementation demonstrates layer-wise pretraining of a deep neural network on the MNIST dataset. The objective is to understand how sequential training of layers affects convergence and performance compared to standard training from scratch.

The network is trained under two conditions:

1. With layer-wise pretraining  
2. Without pretraining  

A comparison is performed to analyze the effectiveness of pretraining.

## Problem Statement

Train a deep neural network with the following specifications:

Input Layer: 28 × 28 flattened to 784  
Hidden Layers:  
First hidden layer: 16 neurons  
Remaining hidden layers: 8 neurons each  
Total hidden layers: 10  
Activation Function: Sigmoid  
Dropout: 0.2  
Output Layer: 10 neurons with Softmax  
Epochs: 10  

## Concept of Unsupervised Pretraining

Unsupervised pretraining is a technique where each layer of a deep neural network is trained sequentially before performing full supervised training.

Instead of initializing all weights randomly, each layer learns meaningful representations from the input data step by step. This improves the initial weight configuration and helps the network converge more effectively.

In this implementation, layers are trained in a greedy layer-wise manner. Each layer is trained using classification loss with respect to ground truth labels while keeping previously trained layers fixed.

The training flow is:

Input → Hidden Layer 1  
Hidden Layer 1 → Hidden Layer 2  
Hidden Layer 2 → Hidden Layer 3  
Hidden Layer 3 → Hidden Layer 4  
Hidden Layer 4 → Hidden Layer 5  
Hidden Layer 5 → Hidden Layer 6  
Hidden Layer 6 → Hidden Layer 7  
Hidden Layer 7 → Hidden Layer 8  
Hidden Layer 8 → Hidden Layer 9  
Hidden Layer 9 → Hidden Layer 10  
Hidden Layer 10 → Output  

Each layer progressively learns higher-level features.

## Methodology

### Layer-wise Pretraining

Train one layer at a time using input data and labels  
Freeze previously trained layers  
Update only the current layer’s weights  
Repeat for all hidden layers  

### Fine Tuning

Unfreeze all layers  
Train the complete network using supervised learning  
Optimize all weights jointly  

### Baseline Model

Same architecture  
Trained from scratch using random initialization  
Used for comparison  

## Architecture

784 → 16 → 8 → 8 → 8 → 8 → 8 → 8 → 8 → 8 → 8 → 10  

Sigmoid activation in hidden layers  
Softmax activation in output layer  
Dropout = 0.2  

## Dataset

MNIST dataset:

60,000 training samples  
10,000 testing samples  
Handwritten digit images (0–9)  

## Training Details

Optimizer: Adam  
Loss Function: Negative Log Likelihood Loss  
Batch Size: 64  
Epochs: 10  

## Results

### Final Comparison

Without Pre-training  
Accuracy: 11.35 %  
Loss: 2.3011  

With Pre-training  
Accuracy: 98.27 %  
Loss: 0.1201  

Overall Improvement in Accuracy: 86.92 %

The results clearly show that layer-wise pretraining significantly improves model performance by enabling better weight initialization and faster convergence.

## Key Observations

Pretraining provides better weight initialization  
Deep networks learn more effectively when trained layer-wise  
Convergence is faster compared to training from scratch  
Final accuracy improves significantly  

## Conclusion

This implementation demonstrates that layer-wise pretraining can drastically enhance the performance of deep neural networks. By training layers sequentially, the model develops meaningful feature representations before full optimization, leading to improved accuracy and stability.



