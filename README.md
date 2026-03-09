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
├── generate_data.py        # Synthetic dataset generation
├── preprocessing.py       # Data cleaning & preprocessing
├── transformer_model.py   # Transformer architecture
├── train.py               # Training pipeline
├── test.py                # Model evaluation
├── main_api.py            # FastAPI service
├── test_api.py            # FastAPI request testing
├── app_streamlit.py       # Streamlit web app
└── requirements.txt       # Dependencies
```

---

## ⚙️ Setup

Clone the repository:

```bash

git clone git@github.com:hasanbahcecii/traffic-volume-forecasting.git
cd traffic-volume-forecasting
```
Create and activate a virtual environment:

```bash

python3 -m venv venv
source venv/bin/activate
```
Install dependencies:

```bash

pip install -r requirements.txt
```

---

## 🚀 Usage

**Data Analysis**

```bash

python load_and_explore.py
```
**Preprocessing**

```bash

python preprocessing.py
```
**Model Training**

```bash

python train.py
```
**Model Testing**

```bash

python test.py
```
**Run FastAPI Service**

```bash

uvicorn main_api:app --reload
```
**Test FastAPI Requests**

```bash

python test_api.py
```
**Launch Streamlit App**

```bash

streamlit run app_streamlit.py
```

---

## 📈 Expected Outcome

- A Transformer-based fraud detection system capable of classifying user behavior into normal, bot, or fraudster categories with high accuracy.
- This approach leverages sequential modeling power of Transformers to capture temporal dependencies in user actions.

---

## 📜 License

MIT
