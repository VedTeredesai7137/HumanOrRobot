```markdown
#Behavioral Bot Detection System

This is a machine learning-based classification tool designed to distinguish between human users and automated bidding bots. Using behavioral patterns from the Facebook Recruiting dataset, the system analyzes bidding frequency, concurrency, and network diversity to identify suspicious activity with high precision.

## 🚀 Features
- **XGBoost Classifier:** Powered by a tuned XGBoost model with a custom decision threshold (0.09) optimized for recall.
- **Real-time GUI:** Built with Python's Tkinter for instant behavioral classification.
- **Advanced Feature Engineering:** Maps simple user inputs to complex behavioral metrics like bidding concurrency, time differentials, and IP rotation ratios.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Machine Learning:** XGBoost, Scikit-Learn
- **Data Handling:** Pandas, NumPy
- **GUI:** Tkinter
- **Model Persistence:** Joblib

## 📋 Requirements
Ensure you have the following dependencies listed in a `requirements.txt` file:
```text
joblib==1.5.3
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
xgboost==3.2.0
```

## 💻 Setup and Execution

### For Windows (Command Prompt / PowerShell)
```powershell
# 1. Create a virtual environment
python -m venv env

# 2. Activate the environment
.\env\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the application
python main.py
```

### For Linux / macOS (Terminal)
```bash
# 1. Create a virtual environment
python3 -m venv env

# 2. Activate the environment
source env/bin/activate

# 3. Install requirements
pip install -r requirements.txt

# 4. Run the application
python3 main.py
```

## 🧪 Recommended Test Cases
Use these values to verify the model's accuracy during demonstrations:

### Test Case 1: The Bot (Expected: BOT DETECTED)
- **Total Bids:** 8500
- **Unique Auctions:** 12
- **Unique IPs:** 25
- **Mean Time Diff:** 0.02
- **Concurrent Bids Ratio:** 0.75

### Test Case 2: The Human (Expected: HUMAN VERIFIED)
- **Total Bids:** 14
- **Unique Auctions:** 3
- **Unique IPs:** 1
- **Mean Time Diff:** 145.0
- **Concurrent Bids Ratio:** 0.0

## 🏗️ Architecture Note
The system uses **Clamping Logic** to prevent extreme input ratios from breaking model logic, ensuring stability even with unrealistic "stress-test" inputs. The current decision threshold is set at **0.09** to prioritize identifying bots in a high-risk environment.

---
**Developer:** Ved Teredesai  
**Department:** Information Technology
```