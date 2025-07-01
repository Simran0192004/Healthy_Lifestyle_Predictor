# 🧠 Healthy Lifestyle Classifier

This project predicts a person's lifestyle category — **Healthy**, **Moderate**, or **Unhealthy** — based on their daily habits such as water intake, sleep, steps walked, junk food score, screen time, and stress levels.

## 📊 Features Used
- Water Intake (liters/day)
- Sleep Hours
- Steps Per Day
- Junk Food Score (0–10)
- Screen Time (hours/day)
- Stress Level (1–10)

## 🚀 Technologies Used
- Python
- Scikit-learn (KNN Classifier)
- Pandas, NumPy
- Matplotlib (optional visualizations)

## 🎯 Model
We use **K-Nearest Neighbors** (KNN) to classify a person’s lifestyle into:
- **Healthy**
- **Moderate**
- **Unhealthy**

## ✅ Results
Achieved 100% accuracy on test data — perfect classification with no mislabels!

## 📁 How to Run

1. Clone this repo or download the files
2. Install the requirements:pip install -r requirements.txt

## ✨ Output Example
```bash
✅ Confusion Matrix:
[[1 0 0]
[0 4 0]
[0 0 1]]

✅ Accuracy Score: 1.0
