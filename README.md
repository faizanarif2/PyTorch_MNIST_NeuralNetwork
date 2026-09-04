This simple practice project demonstrates how to build and train a deep learning model.
This repository contains three complete implementations for classifying handwritten digits from the MNIST dataset: a manual Raw PyTorch approach, a structured PyTorch Lightning approach, and one using CNN.

It serves as a direct comparison between building deep learning pipelines from scratch and using modern production-grade frameworks.

-------------------------------

## 🛠 Tech Stack & Tools

- **Core Frameworks:** PyTorch (`torch`, `torchvision`), PyTorch Lightning (`pytorch_lightning`)
- **Model Architectures:** Multi-Layer Perceptrons (`nn.Linear`), Convolutional Neural Networks (`nn.Conv2d`, `nn.MaxPool2d`, `nn.BatchNorm2d`)
- **Pipeline Components:** `nn.Module`, `LightningModule`, `LightningDataModule`
- **Automation & Tracking:** Lightning Callbacks (`ModelCheckpoint`, `EarlyStopping`), `Trainer`

-------------------------------

### 1. Raw PyTorch Implementation 
Demonstrates manual deep learning fundamentals from the ground up:
- **Data Pipeline:** Direct batching with `torch.utils.data.DataLoader`.
- **Model Architecture:** Fully connected neural network (`nn.Module`) with Linear layers and ReLU activations.
- **Manual 5-Step Training Loop:**
  1. Forward Pass (`pred = model(xb)`)
  2. Loss Computation (`loss = loss_fn(pred, yb)`)
  3. Gradient Reset (`opt.zero_grad()`)
  4. Backpropagation (`loss.backward()`)
  5. Weight Updates (`opt.step()`)

### 2. PyTorch Lightning Implementation 
Refactors the project into modular, production-grade code:
- **`LightningDataModule`:** Isolates downloading, transforms, and DataLoaders into a reusable pipeline.
- **`LightningModule`:** Encapsulates the network architecture, loss function, `training_step`, and `validation_step`.
- **`Trainer` Automation:** Eliminates boilerplate training loops, handles hardware selection (`accelerator="auto"`), and automates metric tracking.
- **Production Callbacks:** Uses `ModelCheckpoint` to save optimal weights based on `val_loss` and `EarlyStopping` to prevent overfitting.

### 3. PyTorch Lightning CNN & Sequence Model 
Upgrades the network architecture to handle 2D spatial features and multi-character text lines:
- **Spatial Feature Extraction:** Uses 2D Convolutional layers (`nn.Conv2d`), Batch Normalization (`nn.BatchNorm2d`), and Max Pooling (`nn.MaxPool2d`) to extract visual features like curves and edges.
- **Synthetic Sequence Pipeline:** Extends single-digit data into multi-character sequence inputs using custom synthetic dataset generators (`MNISTLinesDataModule`).
- **Parallel Classification Head:** Reshapes feature maps to predict multiple character positions simultaneously across sequence images.


