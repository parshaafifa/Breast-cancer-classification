# Breast-cancer-classification
Breast cancer tumor classification using deep learning
# Tumor Classification Using Machine Learning

This project classifies breast cancer tumors as **Malignant** or **Benign** using machine learning. The model takes 30 input features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## 🚀 Project Overview

- **Input**: 30 real-valued features
- **Output**: Predicted tumor class (0 = Malignant, 1 = Benign)
- **Model**: Pre-trained classifier (e.g., Neural Network or Random Forest)
- **Tools**: Python, NumPy, Scikit-learn, TensorFlow/Keras (optional)

---

## 📂 Files Included

| File | Description |
|------|-------------|
| `predict.py` | Main script to predict tumor class from input data |
| `model.h5` or `model.pkl` | Pre-trained model file |
| `scaler.pkl` | StandardScaler object to preprocess input |
| `requirements.txt` | List of Python packages needed |
| `README.md` | Project documentation |

---

## 📊 Example Input

```python
input_data = [15.34,14.26,102.5,704.4,0.1073,...,0.09946]
