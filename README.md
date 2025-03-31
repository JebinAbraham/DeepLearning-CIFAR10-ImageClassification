# CIFAR-10: Model Comparison (FFN, RNN, LSTM, Transformer)

This project explores and compares the performance of different neural network architectures on the CIFAR-10 image classification task using PyTorch. The following models are implemented and evaluated:

- Feedforward Neural Network (FFN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Vision Transformer (ViT)

---

## 📦 Dataset
**CIFAR-10**: 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Split into 50,000 training and 10,000 test images.

Images are resized to 224x224 for compatibility with the Vision Transformer.

---

## 🧠 Models

### 1. Feedforward Neural Network (FFN)
- Input: Flattened 224x224x3
- 2 hidden layers: 1024, 512 units
- Activation: ReLU
- Output: 10 units (softmax via CrossEntropyLoss)

### 2. Recurrent Neural Network (RNN)
- Input: Reshaped as sequence of 32 steps, 224x3 features
- RNN Layer: hidden size 512
- Output: final time-step to classification layer

### 3. Long Short-Term Memory (LSTM)
- Same as RNN, but uses LSTM instead
- Better captures long-term dependencies

### 4. Vision Transformer (ViT)
- Predefined `vit_b_16` architecture from torchvision
- Replaces classification head with a 10-class output
- Input resized to 224x224

---

## ⚙️ Training & Evaluation

- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epochs: 1 (for quick comparison; can be increased)

Metrics:
- Training Loss & Accuracy
- Test Loss & Accuracy
- Classification Report (Precision, Recall, F1)
- Confusion Matrix
- Sample Predictions Visualization

---

## 📊 Results Summary

| Model       | Train Acc | Test Acc | Train Loss | Test Loss |
|-------------|-----------|----------|------------|-----------|
| FFN         | xx%       | xx%      | xx.x       | xx.x      |
| RNN         | xx%       | xx%      | xx.x       | xx.x      |
| LSTM        | xx%       | xx%      | xx.x       | xx.x      |
| Transformer | xx%       | xx%      | xx.x       | xx.x      |

> Replace "xx" with actual results after training

---

## 📈 Visuals
- Bar chart comparison of accuracy & loss
- Confusion matrix heatmaps for each model
- Sample images with ground truth and predicted labels

---

## 🔧 Fine-Tuning (Next Steps)
- Add dropout, batch norm, data augmentation
- Use learning rate schedulers (e.g. StepLR, CosineAnnealing)
- Increase training epochs
- Use pre-trained weights for ViT

---

## 💻 Requirements
```bash
pip install torch torchvision matplotlib scikit-learn seaborn
```

Run the full notebook in [Google Colab](https://colab.research.google.com/) for GPU support.

---

## 🧪 Authors
Built as part of a research assignment on comparing model architectures on CIFAR-10.

Feel free to fork and adapt this template for your own projects!

