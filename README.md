This simple practice project demonstrates how to build and train a deep learning model.
This repository contains two complete implementations for classifying handwritten digits from the MNIST dataset: a manual Raw PyTorch approach and a structured PyTorch Lightning one.

It serves as a direct comparison between building deep learning pipelines from scratch and using modern production-grade frameworks.

-------------------------------

## 🛠 Tech Stack & Tools

- **Core Frameworks:** PyTorch (`torch`, `torchvision`), PyTorch Lightning (`pytorch_lightning`)
- **Model Components:** `nn.Module`, `LightningModule`, `LightningDataModule`
- **Visualization & Logging:** Matplotlib, Lightning Callbacks (`ModelCheckpoint`, `EarlyStopping`)

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
