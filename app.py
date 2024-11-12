# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Policy Recommendation System", layout="wide")

# Load the insurance policies dataset
@st.cache
def load_policy_data():
    return pd.read_csv("insurance_policies_dataset.csv")

# Load and preprocess the transaction data
@st.cache
def preprocess_transactions(transactions_df):
    transactions_df['date'] = pd.to_datetime(transactions_df['date'], errors='coerce')
    transactions_df['month'] = transactions_df['date'].dt.month
    monthly_spend = transactions_df.groupby(['user ID', 'month'])['amount'].sum().unstack(fill_value=0)
    category_spend = transactions_df.groupby(['user ID', 'category'])['amount'].sum().unstack(fill_value=0)
    spending_summary = monthly_spend.join(category_spend, how="inner", lsuffix="_monthly", rsuffix="_category")
    return spending_summary

# Function to preprocess the policy data
def preprocess_policy_data(policies_df):
    le_type = LabelEncoder()
    le_liquidity = LabelEncoder()
    
    # Encode categorical columns
    policies_df['Policy Type'] = le_type.fit_transform(policies_df['Policy Type'])
    policies_df['Liquidity'] = le_liquidity.fit_transform(policies_df['Liquidity'])
    
    # Select features and label
    X = policies_df[['Risk Level', 'Expected ROI', 'Investment Horizon', 'Liquidity', 'Minimum Investment', 
                     'Historical 1-Year Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']]
    y = policies_df['Policy Name']
    
    # Check and handle missing or non-numeric values in X
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
    if X.isnull().values.any():
        X = X.fillna(0)  # Fill any NaN values that remain with 0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, le_type, le_liquidity

# Train the recommendation model
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Training Accuracy: {accuracy:.2f}")
    return model

# Load datasets and preprocess
policies_df = load_policy_data()
X_scaled, y, scaler, le_type, le_liquidity = preprocess_policy_data(policies_df)
model = train_model(X_scaled, y)

# Save the trained model and preprocessing objects for reuse
joblib.dump(model, "policy_recommendation_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_type, "le_type.pkl")
joblib.dump(le_liquidity, "le_liquidity.pkl")

# Streamlit Interface
st.title("Insurance Policy Recommendation System")
st.header("User Profile and Preferences")

# User Inputs
risk_level = st.slider("Select your Risk Level (1 to 5)", min_value=1, max_value=5)
investment_horizon = st.slider("Investment Horizon (years)", min_value=1, max_value=30)
expected_roi = st.number_input("Expected ROI (%)", min_value=0.0, max_value=100.0)
liquidity_pref = st.selectbox("Liquidity Preference", ["Low", "Medium", "High"])

uploaded_file = st.file_uploader("Upload your transaction file (CSV)", type=["csv"])
if uploaded_file is not None:
    transactions_df = pd.read_csv(uploaded_file)
    spending_summary = preprocess_transactions(transactions_df)
    st.write("Spending Summary:")
    st.write(spending_summary)

# Preprocess user input for prediction
def process_user_inputs(risk_level, investment_horizon, expected_roi, liquidity_pref, scaler, le_liquidity):
    liquidity_encoded = le_liquidity.transform([liquidity_pref])[0]
    input_features = np.array([[risk_level, expected_roi, investment_horizon, liquidity_encoded, 
                                spending_summary.sum().sum() if uploaded_file else 0,
                                0, 0, 0, 0]])  # placeholders for other features
    input_features_scaled = scaler.transform(input_features)
    return input_features_scaled

# Make recommendations
if st.button("Get Policy Recommendations"):
    input_features_scaled = process_user_inputs(risk_level, investment_horizon, expected_roi, 
                                                liquidity_pref, scaler, le_liquidity)
    
    model = joblib.load("policy_recommendation_model.pkl")
    recommended_policy = model.predict(input_features_scaled)[0]
    
    st.subheader("Recommended Policy:")
    st.write(recommended_policy)
    
    # Show policy details and comparisons
    recommended_details = policies_df[policies_df['Policy Name'] == recommended_policy]
    st.write("Policy Details:")
    st.write(recommended_details)
    
    # Comparative Visualization for All Policies
    st.subheader("Comparative Study of Recommended Policy vs Others")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Bar Chart: Comparison of Expected ROI
    sns.barplot(data=policies_df, x='Policy Name', y='Expected ROI', ax=ax[0], palette="Blues")
    ax[0].set_title("Expected ROI Comparison")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
    ax[0].bar_label(ax[0].containers[0])

    # Highlight the recommended policy
    recommended_index = policies_df[policies_df['Policy Name'] == recommended_policy].index[0]
    ax[0].patches[recommended_index].set_facecolor("orange")

    # Line Chart: Volatility vs. Historical Return
    sns.scatterplot(data=policies_df, x='Volatility', y='Historical 1-Year Return', hue='Policy Name', ax=ax[1])
    ax[1].set_title("Volatility vs. Historical Return")
    ax[1].legend(loc='best')

    st.pyplot(fig)

    # Pie Chart: Spending Distribution (if transactions uploaded)
    if uploaded_file is not None:
        st.subheader("Spending Distribution by Category")
        category_totals = transactions_df.groupby("category")["amount"].sum()
        fig2, ax2 = plt.subplots()
        ax2.pie(category_totals, labels=category_totals.index, autopct='%1.1f%%', startangle=140)
        ax2.set_title("Spending by Category")
        st.pyplot(fig2)
    
    # Show additional information about profitability, volatility, etc.
    st.write("This policy offers a balance between expected ROI and volatility, "
             "making it suitable for your selected risk tolerance and liquidity preference.")
