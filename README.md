# Face Mask Detection

This project focuses on detecting whether individuals are wearing face masks using both classical machine learning and deep learning techniques. Motivated by the COVID-19 pandemic, this work aims to contribute to public health and safety by identifying unmasked faces in images.

---

## 🎯 Goal

To build a computationally efficient tool that detects people not wearing a face mask using:
- Classical ML (Logistic Regression, MLP)
- Deep learning with transfer learning (DenseNet121)

---

## 🧠 Methods

### 📊 Data Preparation
- Annotated datasets from [Kaggle](https://www.kaggle.com/) including:
  - Face location and mask status labels
  - Cropping and resizing faces for consistency
  - Feature vector generation for ML models

### ⚙️ Models Implemented
1. **Logistic Regression**:
   - Baseline for binary classification
   - Lower F1-score on "No Mask" class

2. **MLP (Multi-Layer Perceptron)**:
   - Improved architecture over logistic regression
   - Better performance on both mask/no-mask classes

3. **CNN with Transfer Learning**:
   - Based on `DenseNet121`
   - Achieved the highest F1-score and robustness on diverse mask types

---

## 📊 Model Performance

| Model               | Class     | Precision | Recall | F1-Score |
|--------------------|-----------|-----------|--------|----------|
| Logistic Regression| Mask      | 0.55      | 0.88   | 0.68     |
|                    | No Mask   | 0.64      | 0.23   | 0.34     |
| MLP                | Mask      | 0.87      | 0.54   | 0.66     |
|                    | No Mask   | 0.67      | 0.92   | 0.78     |
| CNN (DenseNet121)  | Mask      | 0.99      | 0.68   | 0.81     |
|                    | No Mask   | 0.93      | 0.99   | 0.96     |

---

## 📁 Files

- `Mask_Pytorch_Transfer_learning.ipynb` – Transfer learning using DenseNet121
- `Mask__MLP.ipynb` – MLP and logistic regression with feature vectors
- `Model_Performance.png`, `Results.png` – Plots and confusion matrices
- `Face_mask_recognition.pdf` – Full project report with methodology and results

---

## 🔧 Technologies Used

- Python, PyTorch
- Scikit-learn, NumPy, OpenCV
- Jupyter Notebook

---

## 🚀 How to Run

```bash
git clone https://github.com/HabiAshourichoshali/Face_Mask_Detection.git
cd Face_Mask_Detection
pip install -r requirements.txt  # (Create if not already there)
jupyter notebook Mask_Pytorch_Transfer_learning.ipynb
