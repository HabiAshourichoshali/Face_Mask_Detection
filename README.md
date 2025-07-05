# Face Mask Detection

This project focuses on detecting whether individuals are wearing face masks using both classical machine learning (MLP) and deep learning (CNN with transfer learning). It demonstrates the development and comparison of different modeling strategies for a real-world classification task.

---

## 🎯 Objective

To develop models that can classify face images as:
- Wearing a mask (`face_mask`)
- Not wearing a mask (`face_no_mask`)

---

## 🧠 CNN Model with Transfer Learning (PyTorch)

The `Mask_Pytorch_Transfer_learning.ipynb` notebook implements a deep learning pipeline for face mask detection using **PyTorch** and **transfer learning**. A pre-trained CNN model (e.g., DenseNet121) is fine-tuned on a labeled dataset of masked and unmasked face images.

Key components:
- Uses transfer learning to leverage pre-trained image features
- Applies data augmentation and normalization
- Trains and evaluates on a custom image dataset from Google Drive
- Achieves high accuracy (~97%) with robust generalization, even on unseen test images

This model significantly outperforms classical baselines and demonstrates the strength of deep learning for image classification.

---

## 📦 Classical MLP Model (Baseline)

The `Mask__MLP.ipynb` notebook implements a basic image classification pipeline using a **Multi-Layer Perceptron (MLP)** model. Face images are resized, flattened into feature vectors, and used to train a classifier that predicts whether the person is wearing a mask. While less powerful than CNN-based approaches, this notebook demonstrates a classical machine learning baseline for comparison.

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

- `Mask_Pytorch_Transfer_learning.ipynb`: Transfer learning using PyTorch
- `Mask__MLP.ipynb`: Classical ML (MLP baseline)
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
pip install -r requirements.txt  # Add if using pip environment
jupyter notebook
