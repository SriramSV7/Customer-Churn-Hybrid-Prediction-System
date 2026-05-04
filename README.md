📌 Overview

This project is an end-to-end machine learning system that predicts customer churn by combining structured behavioral data with unstructured text data (customer feedback).

It uses a hybrid approach by integrating Sentiment Analysis (NLP) with traditional features, improving predictive performance and making the model more realistic for real-world scenarios.

🚀 Key Highlights
Hybrid Modeling Approach
Combines tabular features (tenure, contract type, charges) with a SentimentScore derived from customer feedback using VADER.

Feature Engineering with NLP
Extracted sentiment polarity from text data to capture customer satisfaction signals.

Handling Class Imbalance
Applied SMOTE to balance churn vs non-churn classes.

Model Selection
Used XGBoost, which performs well on structured data and handles non-linearity effectively.

End-to-End Pipeline
Includes preprocessing pipeline (preprocessor.pkl) and trained model (churn_model.pkl) for consistent inference.

Deployment Ready
Built a Streamlit web app for real-time churn prediction.

🛠️ Tech Stack
Data Processing: Pandas, NumPy
NLP: NLTK (VADER Sentiment Analysis)
Machine Learning: XGBoost, Scikit-learn
Deployment: Streamlit, Joblib

📊 Model Performance
Metric	Score
Accuracy	82.19%
Precision (Churn)	0.65
Recall (Churn)	0.72
F1 Score (Churn)	0.68

👉 The model achieves balanced performance, especially improving recall for churn prediction.

🧩 How It Works
User inputs customer details + optional feedback
Feedback is processed using VADER → SentimentScore
Data is passed through preprocessing pipeline
Model predicts churn probability

📂 Project Structure
Customer_Churn.ipynb → Data cleaning, feature engineering, SMOTE, model training
churn_app.py → Streamlit app for real-time prediction
preprocessor.pkl → Fitted preprocessing pipeline
churn_model.pkl → Trained XGBoost model

🎯 Key Insight

Customer sentiment plays a significant role in churn prediction.
Incorporating NLP features improved the model’s ability to capture customer dissatisfaction patterns.
