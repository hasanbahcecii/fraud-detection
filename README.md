# Fraud Detection with Transformers

## 📌 Problem Definition
The goal of this project is to detect fraudulent behavior on an e-commerce platform using a **Transformer-based model**.  
We aim to classify users during their sessions as either:
- **Normal users**
- **Bots**
- **Fraudsters**

### User Actions Considered
- Number of clicks  
- Time spent on pages  
- Amount spent  
- Scroll depth  
- Cart events  

By analyzing these behavioral features across time steps, the model will learn to distinguish between legitimate and malicious activity.

---

## 📊 Dataset
- **Synthetic dataset** with sequential user actions  
- **20 time steps** per user session  
- Features:
  - `click_duration`
  - `page_duration`
  - `amount_spent`
  - `scroll_depth`
  - `cart_events`
- **Classes**:
  - `0` → Normal user  
  - `1` → Bot  
  - `2` → Fraudster  
- **Size**: ~500 samples per class, total **1500 samples**

---

## 🛠️ Technologies
- **PyTorch** (Transformer implementation)  
- **NumPy, Pandas** (data handling)  
- **Matplotlib, Seaborn** (visualization)  
- **Scikit-learn** (preprocessing, evaluation)  
- **TQDM** (progress tracking)

---

## 📂 Project Structure
```
fraud-detection-transformer/
│
├── generatedata.py        # Synthetic dataset generation
├── preprocessing.py       # Data cleaning & preprocessing
├── transformer_model.py   # Transformer architecture
├── train.py               # Training pipeline
├── test.py                # Model evaluation
└── requirements.txt       # Dependencies
```

---

## ⚙️ Installation
To set up the environment, install the required libraries:

```bash
pip install numpy pandas matplotlib scikit-learn torch tqdm seaborn
```

---

## 🚀 Plan

1. Data Generation → Create synthetic dataset with user actions.

2. Preprocessing → Normalize, encode, and prepare sequences.

3. Modeling → Implement a Transformer-based classifier in PyTorch.

4. Training → Train the model on the dataset.

5. Testing → Evaluate performance on unseen samples.

---

## 📈 Expected Outcome

A Transformer-based fraud detection system capable of classifying user behavior into normal, bot, or fraudster categories with high accuracy.
This approach leverages sequential modeling power of Transformers to capture temporal dependencies in user actions.