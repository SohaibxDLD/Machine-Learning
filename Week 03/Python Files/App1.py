import streamlit as st
import pandas as pd
import requests

st.title("Mall Customers Clustering & Classification")

st.header("Upload Customer Dataset")
uploaded_file = st.file_uploader("Choose CSV file", type="csv")
backend_url = "http://localhost:8000"

if uploaded_file:
    st.write("Preview:")
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    if st.button("Run Clustering & Analysis"):
        resp = requests.post(
            f"{backend_url}/analyze/",
            files={"file": uploaded_file.getvalue()}
        )
        if resp.ok:
            result = resp.json()
            st.success(result["message"])
            st.write(f"Optimal clusters: {result['optimal_clusters']}")
            st.table(pd.DataFrame([{
                "Cluster Label": label, "Count": count
            } for label, count in result['cluster_counts'].items()]))
            st.write("Cluster Identities:", result["cluster_labels"])
        else:
            st.error("Analysis failed.")

st.header("Predict Cluster for New Customer")
with st.form("predict"):
    genre = st.selectbox("Gender (encoded: 0=Female, 1=Male)", [0, 1])
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    income = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=50)
    score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    submit = st.form_submit_button("Predict Cluster")
    if submit:
        data = {
            "Genre": genre,
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": score
        }
        pred = requests.post(f"{backend_url}/predict/", json=data)
        if pred.ok:
            st.write(
                f"Predicted Cluster: {pred.json()['cluster']} "
                f"({pred.json().get('label', 'Unknown')})"
            )
        else:
            st.error(pred.json().get("error", "Prediction failed."))
