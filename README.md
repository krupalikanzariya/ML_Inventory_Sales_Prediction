# 📦 Inventory Demand Prediction

## 🔍 Overview
This project predicts inventory demand using Machine Learning, specifically an **XGBoost model** trained on historical sales data. The application is built with **Flask** for the backend and deployed on **Render**. The model achieves an **accuracy of 0.92**.

## 🚀 Features
- 📅 Predict demand based on date, store ID, and item ID.
- 🧠 Uses **XGBoost**, a powerful gradient boosting algorithm.
- 🌐 Flask web application with an interactive UI.
- 🌍 Public deployment using Render.

## 📌 Tech Stack
- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS (Bootstrap)
- **Model**: XGBoost
- **Deployment**: Render, GitHub Pages

## 📂 Installation
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/krupalikanzariya/ML_Inventory_Sales_Prediction.git
cd ML_Inventory_Sales_Prediction
```

### **2️⃣ Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Flask App Locally**
```bash
python app.py
```

Access the app in your browser at **`http://127.0.0.1:5000/`**.

## 📦 Libraries Used
- `Flask` - Web framework for Python.
- `joblib` - Model loading.
- `pandas` - Data manipulation.
- `numpy` - Numerical computations.
- `xgboost` - Machine learning model.
- `datetime` - Date handling.
- `holidays` - Checking holiday dates.

## 📊 Model Accuracy
The XGBoost model achieves an **accuracy of 0.92** on test data.
