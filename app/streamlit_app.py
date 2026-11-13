
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CHURNIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "customer_churn_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "churn_threshold.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")


# =========================================================
# 2. PROFESSIONAL DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F4F7FC;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #111D35;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: transparent;
    padding: 9px;
    border-radius: 8px;
}

h1 {
    color: #14213D;
    font-weight: 800;
    letter-spacing: -1px;
}

h2, h3 {
    color: #1E293B;
    font-weight: 700;
}

p {
    color: #64748B;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E6EBF2;
    padding: 19px;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #14213D;
    font-weight: 800;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #2563EB;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 45px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #1D4ED8;
    color: white;
    border: none;
}

[data-testid="stForm"] {
    background: white;
    padding: 22px;
    border: 1px solid #E5EAF2;
    border-radius: 15px;
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

.hero {
    background: linear-gradient(120deg, #101D3A 0%, #1E40AF 65%, #2563EB 100%);
    padding: 35px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.hero p {
    color: #DCE8FF;
    font-size: 15px;
}

.section-label {
    color: #2563EB;
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.info-card {
    background: white;
    border: 1px solid #E5EAF2;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
}

.info-card h3 {
    font-size: 17px;
    margin-bottom: 8px;
}

.info-card p {
    font-size: 13px;
    margin-bottom: 0;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. LOAD MODEL AND DATA
# =========================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, threshold


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    ).fillna(0)

    return data


try:
    model, threshold = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div style="padding:12px 5px 22px 5px;">
    <h1 style="color:white;font-size:29px;margin:0;">
        CHURNIQ<span style="color:#60A5FA;">.</span>
    </h1>
    <p style="color:#AFC2E2;font-size:11px;letter-spacing:2px;">
        CUSTOMER INTELLIGENCE
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Executive Overview",
        "📊 Customer Analytics",
        "🤖 AI Prediction Studio",
        "🧠 Model Intelligence",
        "💡 Retention Strategy"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("### MODEL STATUS")
st.sidebar.success("Model loaded successfully")

st.sidebar.markdown("**Algorithm:** Logistic Regression")
st.sidebar.markdown(f"**Threshold:** {threshold:.2f}")

st.sidebar.markdown("---")
st.sidebar.caption("CHURNIQ | Data Science Portfolio Project")
st.sidebar.caption("IBM Telco Customer Churn Dataset")


# =========================================================
# 5. REUSABLE HEADER
# =========================================================

def page_header(title, subtitle):
    st.markdown(f"""
    <div style="margin-bottom:25px;">
        <div class="section-label">CHURNIQ / CUSTOMER INTELLIGENCE</div>
        <h1 style="margin-top:8px;">{title}</h1>
        <p style="font-size:15px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def metric_card(label, value, description):
    st.markdown(f"""
    <div class="info-card">
        <p style="font-size:12px;font-weight:700;letter-spacing:1px;">
            {label}
        </p>
        <h2 style="font-size:29px;margin:8px 0;color:#14213D;">
            {value}
        </h2>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# 6. EXECUTIVE OVERVIEW
# =========================================================

if page == "🏠 Executive Overview":

    page_header(
        "Executive Overview",
        "A data-driven view of customer retention and churn performance."
    )

    st.markdown("""
    <div class="hero">
        <div style="font-size:12px;letter-spacing:2px;color:#BFDBFE;">
            AI-POWERED CUSTOMER INTELLIGENCE PLATFORM
        </div>
        <h1>Understand. Predict. Retain.</h1>
        <p>
            Identify customers at risk, understand churn patterns,
            and support smarter customer retention decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = load_data()

        total = len(df)
        churned = int((df["Churn"] == "Yes").sum())
        retained = int((df["Churn"] == "No").sum())
        churn_rate = churned / total * 100
        retention_rate = retained / total * 100

        st.subheader("Business Performance")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("TOTAL CUSTOMERS", f"{total:,}", "Customers analyzed")

        with c2:
            metric_card("CHURNED CUSTOMERS", f"{churned:,}", "Customers who left")

        with c3:
            metric_card("RETENTION RATE", f"{retention_rate:.2f}%", "Customers retained")

        with c4:
            metric_card("CHURN RATE", f"{churn_rate:.2f}%", "Overall churn percentage")

        st.markdown("---")

        left, right = st.columns(2)

        with left:
            st.subheader("Customer Churn Distribution")
            counts = df["Churn"].value_counts()
            st.bar_chart(counts)

        with right:
            st.subheader("Churn by Contract Type")

            contract_rate = (
                df.groupby("Contract")["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.bar_chart(contract_rate)

        st.markdown("---")

        st.subheader("Key Business Observations")

        a, b, c = st.columns(3)

        with a:
            st.markdown("""
            <div class="info-card">
                <h3>📄 Contract Risk</h3>
                <p>Month-to-month customers show a substantially higher churn rate.</p>
            </div>
            """, unsafe_allow_html=True)

        with b:
            st.markdown("""
            <div class="info-card">
                <h3>🌐 Internet Services</h3>
                <p>Fiber optic customers have higher observed churn than DSL customers.</p>
            </div>
            """, unsafe_allow_html=True)

        with c:
            st.markdown("""
            <div class="info-card">
                <h3>💳 Payment Behavior</h3>
                <p>Electronic check customers show a higher observed churn rate.</p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load business analytics: {e}")


# =========================================================
# 7. CUSTOMER ANALYTICS
# =========================================================

elif page == "📊 Customer Analytics":

    page_header(
        "Customer Analytics",
        "Explore customer behavior, service usage and churn patterns."
    )

    try:
        df = load_data()

        st.subheader("Explore Customer Data")

        selected_contract = st.selectbox(
            "Filter by Contract",
            ["All"] + sorted(df["Contract"].dropna().unique().tolist())
        )

        filtered_df = df.copy()

        if selected_contract != "All":
            filtered_df = filtered_df[
                filtered_df["Contract"] == selected_contract
            ]

        st.caption(f"Showing {len(filtered_df):,} customers")
        st.dataframe(filtered_df.head(100), width="stretch")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Churn by Internet Service")

            internet_rate = (
                df.groupby("InternetService")["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.bar_chart(internet_rate)

        with col2:
            st.subheader("Churn by Payment Method")

            payment_rate = (
                df.groupby("PaymentMethod")["Churn"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .sort_values(ascending=False)
            )

            st.bar_chart(payment_rate)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Tenure Distribution")
            st.histogram if False else None
            st.bar_chart(
                df.groupby("tenure").size().rename("Customers")
            )

        with col4:
            st.subheader("Monthly Charges by Churn")
            monthly = df.groupby("Churn")["MonthlyCharges"].mean()
            st.bar_chart(monthly)

        st.caption(
            "These charts show observed associations, not proof of causation."
        )

    except Exception as e:
        st.error(f"Customer analytics failed: {e}")


# =========================================================
# 8. AI PREDICTION STUDIO
# =========================================================

elif page == "🤖 AI Prediction Studio":

    page_header(
        "AI Prediction Studio",
        "Estimate customer churn probability using the trained machine learning model."
    )

    st.info(
        "Enter customer details below. The model will estimate the probability "
        "of churn and suggest a retention action."
    )

    with st.form("prediction_form"):

        st.subheader("01 | Customer Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])

        with col2:
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone = st.selectbox("Phone Service", ["Yes", "No"])

        with col3:
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

        st.markdown("---")
        st.subheader("02 | Service Subscriptions")

        col4, col5, col6 = st.columns(3)

        with col4:
            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

        with col5:
            protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

        with col6:
            tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )
            movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

        st.markdown("---")
        st.subheader("03 | Billing & Contract")

        col7, col8 = st.columns(2)

        with col7:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col8:
            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                max_value=10000.0,
                value=840.0,
                step=1.0
            )

        submitted = st.form_submit_button(
            "🚀 Predict Customer Churn",
            width="stretch"
        )

    if submitted:

        customer_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges
        }])

        try:
            probability = model.predict_proba(customer_data)[0][1]

            prediction = (
                "Churn" if probability >= threshold else "Not Churn"
            )

            st.markdown("---")
            st.subheader("Prediction Results")

            result1, result2 = st.columns(2)

            with result1:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with result2:
                st.metric(
                    "Decision Threshold",
                    f"{threshold * 100:.0f}%"
                )

            st.progress(float(probability))

            if probability >= 0.70:

                risk = "HIGH RISK"
                st.error("🔴 High Churn Risk")
                recommendation = (
                    "Prioritize this customer for retention outreach. "
                    "Review service concerns and consider a personalized offer."
                )

            elif probability >= threshold:

                risk = "MEDIUM RISK"
                st.warning("🟠 Medium Churn Risk")
                recommendation = (
                    "Monitor this customer. Consider proactive support "
                    "and check for billing or service issues."
                )

            else:

                risk = "LOW RISK"
                st.success("🟢 Low Churn Risk")
                recommendation = (
                    "Continue regular engagement and maintain service quality."
                )

            st.markdown(f"### Customer Risk: {risk}")
            st.info(f"💡 **Retention Recommendation:** {recommendation}")

            st.caption(
                "Risk categories are business heuristics. "
                "The model score is an estimate, not a guarantee of churn."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# =========================================================
# 9. MODEL INTELLIGENCE
# =========================================================

elif page == "🧠 Model Intelligence":

    page_header(
        "Model Intelligence",
        "Evaluate model performance and understand the factors influencing predictions."
    )

    st.subheader("Final Model Performance")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", "75.30%")

    with c2:
        st.metric("Precision", "52.43%")

    with c3:
        st.metric("Recall", "75.13%")

    with c4:
        st.metric("F1 Score", "61.76%")

    with c5:
        st.metric("ROC-AUC", "84.16%")

    st.caption(
        "Final Logistic Regression test results | "
        "Classification threshold: 0.55"
    )

    st.markdown("---")

    st.subheader("Model Comparison")

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "Accuracy": [0.7381, 0.7601, 0.7999],
        "Precision": [0.5043, 0.5347, 0.6554],
        "Recall": [0.7834, 0.7406, 0.5187],
        "F1 Score": [0.6136, 0.6211, 0.5791],
        "ROC-AUC": [0.8416, 0.8372, 0.8441]
    })

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "ROC-AUC": "{:.2%}"
        }),
        width="stretch",
        hide_index=True
    )

    st.markdown("---")

    # Display saved model graphs

    graph_sections = [
        ("Confusion Matrix", "final_confusion_matrix.png"),
        ("ROC Curve Comparison", "roc_curve_comparison.png"),
        ("Feature Importance", "feature_importance.png"),
        ("SHAP Summary Analysis", "shap_summary_plot.png"),
        ("SHAP Feature Importance", "shap_feature_importance.png"),
        ("Model Performance Comparison", "model_performance_comparison.png")
    ]

    for title, filename in graph_sections:

        image_path = os.path.join(FIGURES_DIR, filename)

        if os.path.exists(image_path):
            st.subheader(title)
            st.image(
                image_path,
                caption=title,
                width="stretch"
            )
            st.markdown("---")
        else:
            st.warning(f"{filename} not found.")


# =========================================================
# 10. RETENTION STRATEGY
# =========================================================

elif page == "💡 Retention Strategy":

    page_header(
        "Retention Strategy",
        "Turn churn insights into practical customer retention actions."
    )

    st.subheader("Recommended Business Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🔴 High-Risk Customers</h3>
            <p>
                • Prioritize personalized retention outreach<br>
                • Investigate service complaints<br>
                • Review pricing and suitable offers<br>
                • Follow up on unresolved issues
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>📄 Contract Strategy</h3>
            <p>
                • Encourage suitable long-term plans<br>
                • Explain contract benefits clearly<br>
                • Review month-to-month customer concerns
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🟠 Medium-Risk Customers</h3>
            <p>
                • Monitor customer engagement<br>
                • Offer proactive technical support<br>
                • Check billing experience<br>
                • Track changes in customer behavior
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h3>🟢 Low-Risk Customers</h3>
            <p>
                • Maintain service quality<br>
                • Encourage customer loyalty<br>
                • Continue regular engagement<br>
                • Monitor future risk changes
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Important Business Consideration")

    st.info(
        "A churn prediction model should support retention teams, "
        "not automatically decide customer treatment. "
        "Business teams should validate offers, costs and customer needs."
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div style="text-align:center;padding:15px;">
    <h3 style="color:#1E40AF;">CHURNIQ.</h3>
    <p style="font-size:12px;">
        AI-Powered Customer Churn Prediction & Retention Analytics
    </p>
    <p style="font-size:11px;">
        Built with Python · Streamlit · Scikit-learn · SHAP
    </p>
</div>
""", unsafe_allow_html=True)
