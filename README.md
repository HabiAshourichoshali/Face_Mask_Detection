# Face Mask Detection

This project focuses on detecting whether individuals are wearing face masks using machine learning and deep learning models. Motivated by the COVID-19 pandemic, this tool aims to support public health monitoring through automated image analysis.

---

## 🎯 Objective

To develop a computationally efficient model that detects unmasked individuals using:
- Classical ML: Logistic Regression and Multi-Layer Perceptron (MLP)
- Deep Learning: Transfer learning with DenseNet121

---

## 🧠 Methods

### 🗃️ Data Preparation
- Annotated face-mask datasets from Kaggle
- Cropped and resized face regions
- Flattened feature vectors for classical ML models

### 🧪 Models Implemented
1. **Logistic Regression** – Baseline binary classifier
2. **MLP (Multi-Layer Perceptron)** – Improved architecture
3. **Transfer Learning (DenseNet121)** – High-accuracy deep model

---

## 📊 Model Performance Summary

The model was trained using transfer learning with DenseNet121 and evaluated on a held-out test set.

### 📈 Confusion Matrix & Metrics

![Confusion Matrix and Metrics](Model_Performance.png)

- **Accuracy:** 97%
- **F1-score:** 0.97 for both "Mask" and "No Mask" classes
- Balanced precision and recall across both categories

---

## 🧪 Generalization: Visual Test on Unseen Image

To test real-world performance, the model was evaluated on:
- A sample from the training set
- An unseen image of the author (*not in training data*)

![Generalization Results](Results.png)

**Top Left:** Confusion matrix from DenseNet121  
**Top Right:** Full classification report  
**Bottom Left:** Prediction on a training image  
**Bottom Right:** ✅ Correct prediction on an unseen image of the author — demonstrating strong generalization

---

## 📁 Files

- `Mask_Pytorch_Transfer_learning.ipynb`: Transfer learning using DenseNet121
- `Mask__MLP.ipynb`: Classical ML models (LogReg, MLP)
- `Model_Performance.png`: Confusion matrix + metrics
- `Results.png`: Visual predictions and evaluation
- `Face_mask_recognition.pdf`: Full project report

---

## 🔧 Tools & Technologies

- Python, PyTorch, scikit-learn
- NumPy, OpenCV, Matplotlib
- Jupyter Notebook

---

## 🚀 How to Run

```bash
git clone https://github.com/HabiAshourichoshali/Face_Mask_Detection.git
cd Face_Mask_Detection
pip install -r requirements.txt  # If added
jupyter notebook Mask_Pytorch_Transfer_learning.ipynb
