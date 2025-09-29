import streamlit as st
import pandas as pd
import time
import os
import gdown
import joblib

# --- Download model from Google Drive ---
url = "https://drive.google.com/uc?export=download&id=12_zrTG3H4KICrJKVxC3NPkvyByUbgLiJ"
output = "rf_bank_model.pkl"

if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load trained pipeline
model = joblib.load(output)


# --- Streamlit App Config ---
st.set_page_config(page_title="Bank Term Deposit Predictor", layout="wide")
st.title("💰 Bank Term Deposit Subscription Predictor")

# --- Theme toggle ---
mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        div.stNumberInput > div, div.stSlider > div, div.stButton > button { background-color: #1e1e1e; color: white; }
        label[data-baseweb="label"] { color: white !important; }
        /* Fix numeric input labels */
        div.stNumberInput > label, div.stSlider > label { color: white !important; }
        /* Keep button text default color */
        div.stButton > button { color: initial !important; }
        </style>
    """, unsafe_allow_html=True)


# --- Sidebar for Categorical Inputs ---
st.sidebar.header("Select Categorical Values")
job = st.sidebar.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid",
                                   "management", "retired", "self-employed", "services",
                                   "student", "technician", "unemployed", "unknown"])
marital = st.sidebar.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.sidebar.selectbox("Education Level", ["secondary", "tertiary", "primary", "unknown"])
default = st.sidebar.selectbox("Credit in Default?", ["yes", "no"])
housing = st.sidebar.selectbox("Housing Loan?", ["yes", "no"])
loan = st.sidebar.selectbox("Personal Loan?", ["yes", "no"])
contact = st.sidebar.selectbox("Contact Type", ["cellular", "telephone"])
month = st.sidebar.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun",
                                                    "jul", "aug", "sep", "oct", "nov", "dec"])
poutcome = st.sidebar.selectbox("Previous Campaign Outcome", ["unknown", "success", "failure", "other"])

# --- Placeholder for Prediction Output ---
prediction_placeholder = st.empty()

# --- Form for Numeric Inputs ---
with st.form(key="input_form"):
    st.subheader("Enter Numeric Values")

    age = st.number_input("Age", 18, 100, 30)
    balance = st.number_input("Average Yearly Balance (EUR)", value=1000.0)
    day = st.number_input("Last Contact Day of Month", 1, 31, 15)
    duration = st.number_input("Last Contact Duration (seconds)", value=120)
    campaign = st.number_input("Number of Contacts During Campaign", value=1)
    pdays = st.number_input("Days Since Last Contact", value=-1)
    previous = st.number_input("Previous Contacts", value=0)

    submit_button = st.form_submit_button(label="Predict Subscription")

# --- Optional: age_group function ---
def get_age_group(age):
    if age < 25: return "Young"
    elif age < 35: return "Young Adult"
    elif age < 45: return "Adult"
    elif age < 55: return "Middle-aged"
    elif age < 65: return "Senior"
    else: return "Retired"

age_group = get_age_group(age)

# --- Convert month to numeric ---
month_order = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6,
               'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
month_numeric = month_order[month]

# --- Prediction ---
if submit_button:
    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month_numeric,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'age_group': age_group
    }])

    pred = model.predict(input_df)
    prob = model.predict_proba(input_df)[:, 1]
    result = "✅ Subscribed" if pred[0] == 1 else "❌ Not Subscribed"

    with prediction_placeholder.container():
        st.markdown("### Prediction Result")
        st.success(result)
        st.markdown(f"**Probability of Subscription:** {prob[0]:.2%}")

        probability = prob[0]
        tooltip = "Low → High likelihood of subscription"

        bar_placeholder = st.empty()
        for i in range(0, int(probability*100)+1):
            bar_placeholder.markdown(f"""
                <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px; width: 100%;" title="{tooltip}">
                    <div style="
                        width: {i}%;
                        background: linear-gradient(to right, red, orange, green);
                        height: 25px;
                        border-radius: 5px;
                        text-align: center;
                        color: white;
                        font-weight: bold;">
                        {i}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            time.sleep(0.01)
