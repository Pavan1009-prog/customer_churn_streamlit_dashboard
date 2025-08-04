
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("An interactive dashboard to explore and predict customer churn.")

# Load sample data
@st.cache_data
def load_data():
    df = pd.read_csv("Customer_Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar - Upload new file
st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

# Encode categorical variables for modeling
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    if col != 'customerID':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

# KPIs
total_customers = len(df)
churn_count = df[df['Churn'] == 'Yes'].shape[0]
churn_rate = churn_count / total_customers
retention_rate = 1 - churn_rate

col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2%}")
col3.metric("Retention Rate", f"{retention_rate:.2%}")

# Charts
fig_pie = px.pie(df, names='Churn', title="Churn Distribution", hole=0.4,
                 color='Churn', color_discrete_map={'Yes':'red', 'No':'green'})
fig_bar = px.bar(df.groupby('Contract')['Churn'].value_counts().unstack().fillna(0),
                 title="Contract Type vs Churn", barmode='stack')
fig_box = px.box(df, x='Churn', y='MonthlyCharges', color='Churn',
                 title="Monthly Charges by Churn")

col1, col2 = st.columns(2)
col1.plotly_chart(fig_pie, use_container_width=True)
col2.plotly_chart(fig_bar, use_container_width=True)
st.plotly_chart(fig_box, use_container_width=True)

# Model training and predictions
if 'Churn' in df.columns:
    X = df_encoded.drop(['Churn', 'customerID'], axis=1, errors='ignore')
    y = df_encoded['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Model trained with Accuracy: {accuracy:.2%}")

    # Predict churn for full dataset
    df['Predicted Churn'] = model.predict(X)
    st.subheader("🔍 Predictions Preview")
    st.write(df.head())

    # Download predictions
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "churn_predictions.csv", "text/csv")
else:
    st.error("Dataset must contain a 'Churn' column.")
