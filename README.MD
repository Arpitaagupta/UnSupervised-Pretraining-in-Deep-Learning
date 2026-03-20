# Layer-wise Pretraining in Deep Neural Networks on MNIST

This implementation demonstrates greedy layer-wise pretraining in a deep neural network using the MNIST dataset. The objective is to analyze how sequential training of layers improves model performance compared to training a deep network from scratch.

The experiment follows a structured pipeline where each hidden layer is trained individually, and the learned weights are reused to initialize a deeper network. The final model is then fine-tuned using supervised learning.

A comparison is performed between:

Without pretraining  
With layer-wise pretraining  

## Objective

To implement and evaluate layer-wise pretraining in a deep neural network with the following configuration:

Input Layer: 28 × 28 flattened to 784  

Hidden Layers:
First hidden layer: 512 neurons  
Second and third: 256 neurons  
Fourth to sixth: 128 neurons  
Seventh to ninth: 64 neurons  
Tenth hidden layer: 32 neurons  

Total hidden layers: 10  

Activation Function: Sigmoid  
Dropout: 0.2  
Output Layer: 10 neurons with Softmax  
Epochs:
Pretraining: 10 per layer  
Fine-tuning: 20  
Baseline: 20  

## Dataset

MNIST handwritten digit dataset:

60,000 training samples  
10,000 testing samples  
Grayscale images of size 28 × 28  

## Methodology

### Baseline Model

A deep neural network with 10 hidden layers is trained using random weight initialization. This serves as the reference model.

### Layer-wise Pretraining

Each hidden layer is trained sequentially using the following approach:

Build a partial network up to the current layer  
Train the network using labeled data  
Freeze previously trained layers  
Update only the current layer and output layer  
Store the learned weights  

This process is repeated for all 10 hidden layers.

### Fine Tuning

After pretraining:

All layers are unfrozen  
The complete network is trained jointly  
Weights are refined using backpropagation  

## Architecture

784 → 512 → 256 → 256 → 128 → 128 → 128 → 64 → 64 → 64 → 32 → 10  

Sigmoid activation is used in hidden layers  
Softmax is used in the output layer  
Dropout of 0.2 is applied after each hidden layer  

## Training Details

Optimizer: Adam  
Loss Function: Categorical Crossentropy  
Batch Size: 128  

## Results

### Final Performance Comparison

Without Pre-training  
Accuracy: 11.35 %  
Loss: 2.3011  

With Pre-training  
Accuracy: 98.27 %  
Loss: 0.1201  

Improvement in Accuracy: 86.92 %


## Observations

The model trained without pretraining fails to learn effectively and remains near random guessing.

Layer-wise pretraining enables each layer to learn meaningful representations before full training.

The pretrained model converges faster and achieves significantly higher accuracy.

Weight initialization plays a critical role in training deep neural networks.



## Conclusion

This implementation demonstrates that greedy layer-wise pretraining significantly enhances the performance of deep neural networks. By training layers sequentially, the model develops strong feature representations, leading to improved convergence and accuracy compared to standard training.

