import streamlit as st
import pandas as pd

st.title("Water Quality Monitoring")

uploaded_file = st.file_uploader(
    "Upload your Excel file",
    type=["xlsx", "xls"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.success("File uploaded successfully!")
    st.dataframe(df)
