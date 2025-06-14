# 🍃 Leaf Disease Classification using EfficientNet (PyTorch)

This project implements a deep learning pipeline using EfficientNet (via `efficientnet_pytorch`) to classify diseases in plant leaves. The solution is built using PyTorch and includes dataset preprocessing, training, evaluation, and visualization of performance metrics.

---

## 📁 Project Overview

- **Model**: EfficientNet (transfer learning, PyTorch)
- **Task**: Multi-class classification of plant leaf diseases
- **Input**: Leaf images (RGB)
- **Output**: Disease class label
- **Framework**: PyTorch, torchvision

---

## 🧾 Installation

Before running the code, make sure to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install efficientnet_pytorch torch torchvision matplotlib seaborn pandas scikit-learn tqdm
```

---

## 📦 Dataset Structure

The dataset should be organized as follows:

```
dataset/
├── class_0/
│   ├── image1.jpg
│   └── ...
├── class_1/
│   ├── image2.jpg
│   └── ...
...
```

Each subdirectory corresponds to a different disease class.

---

## 🚀 How to Run

### 1. Train the Model

Run the notebook `LeafClassification_EfficientNet.ipynb` step-by-step:

- Loads and preprocesses image dataset
- Defines custom `Dataset` class with `transforms`
- Uses pretrained `EfficientNet` from `efficientnet_pytorch`
- Trains model and plots training metrics

> Note: Modify paths and parameters in the notebook as needed.

### 2. Evaluate the Model

- Prints accuracy, classification report, and confusion matrix
- Saves the trained model (`.pt` file) for future inference

---

## 🔧 Configuration

Common parameters to tune in the notebook:

- `batch_size`: 16 / 32 / 64 depending on GPU memory
- `img_size`: e.g., 224x224 or 300x300
- `learning_rate`: usually 1e-4 or 1e-5
- `num_epochs`: e.g., 20–50
- `optimizer`: Adam / SGD

---

## 📊 Results

Example results (to be updated after training):

| Metric         | Value     |
|----------------|-----------|
| Accuracy       | 95.2%     |
| F1-score       | 0.951     |
| Inference Time | ~18 ms/img |

> Confusion matrix and classification report will be shown in the notebook.

---

## 📁 Files

```
LeafClassification_EfficientNet/
├── LeafClassification_EfficientNet.ipynb   # Main notebook
├── dataset/                                # Your image dataset
├── model.pt                                # Saved model after training
└── README.md                               # Project description
```

---

## 📜 License

This project is open-source under the MIT License.

---

## 🙌 Acknowledgements

- [EfficientNet (PyTorch)](https://github.com/lukemelas/EfficientNet-PyTorch)
- [PlantVillage Dataset](https://www.kaggle.com/emmarex/plantdisease)
- [PyTorch](https://pytorch.org/)
- [Kaggle community](https://www.kaggle.com/)
