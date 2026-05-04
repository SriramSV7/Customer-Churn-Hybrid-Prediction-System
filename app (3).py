import streamlit as st
import joblib
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Load model and preprocessor
model = joblib.load("churn_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Title
st.title("Customer Churn Prediction App")

st.write("Enter customer details below:")

# ---- USER INPUTS ----
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

feedback = st.text_area("Customer Feedback (Optional)")

# ---- SENTIMENT FUNCTION ----
def get_sentiment_score(text):
    if text:
        return sia.polarity_scores(text)['compound']
    return 0

# ---- HANDLE OPTIONAL FEEDBACK ----
if feedback.strip() != "":
    sentiment_score = get_sentiment_score(feedback)
    st.write(f"Sentiment Score: {sentiment_score}")
else:
    sentiment_score = 0
    st.write("No feedback provided → sentiment = 0")

# ---- PREDICTION ----
if st.button("Predict Churn"):

    if feedback.strip() == "":
        st.warning("⚠️ No feedback provided. Prediction is based on other features.")

    # Create FULL input dataframe (must match training columns)
    input_data = pd.DataFrame([{
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': tenure,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': internet_service,
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': contract,
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': monthly_charges,
        'TotalCharges': monthly_charges * tenure,
        'SentimentScore': sentiment_score
    }])

    try:
        # Transform input
        input_processed = preprocessor.transform(input_data)

        # Handle sparse vs dense safely
        if hasattr(input_processed, "toarray"):
            input_processed = input_processed.toarray()

        # Predict
        prediction = model.predict(input_processed)

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer will stay")

    except Exception as e:
        st.error(f"Error: {e}")