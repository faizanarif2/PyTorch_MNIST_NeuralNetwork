This simple practice project demonstrates how to build and train a deep learning model from scratch, covering manual data loading, network architecture, training loops, and basic evaluation.

1. Data Pipeline: Loading and batching MNIST images using PyTorch `DataLoader`.
2. Model Architecture: Simple neural network with Linear layers and ReLU activations (`nn.Module`).
3. Training Loop: Implementing the 5 PyTorch training steps:
   - Forward pass
   - Calculate loss (`nn.CrossEntropyLoss`)
   - Clear gradients (`optimizer.zero_grad()`)
   - Backpropagation (`loss.backward()`)
   - Update weights (`optimizer.step()`)
  
Used PyTorch,nn.Module,torchvision
