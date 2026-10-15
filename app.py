
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Water Quality Prediction",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Water Quality Monitoring and Prediction")
st.write(
    "Machine Learning-based analysis and future prediction "
    "of Krishna and Panchganga River water quality."
)

st.divider()

# ---------------- EXCEL UPLOAD ----------------
excel_file = "water_quality_data.xlsx"

try:
    df = pd.read_excel(excel_file)
except Exception as e:
    st.error(f"Unable to read Excel file: {e}")
    st.stop()

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# Find Date column, ignoring capitalization and spaces
date_col = next(
    (col for col in df.columns if col.strip().lower() == "date"),
    None
)

if date_col is None:
    st.error("Date column not found in the Excel file.")
    st.write("Available columns:", df.columns.tolist())
    st.info("Rename the date column to Date in your Excel file.")
    st.stop()

# Standardize Date column
df = df.rename(columns={date_col: "Date"})
df["Date"] = pd.to_datetime(
    df["Date"], errors="coerce", dayfirst=True
)
df = df.dropna(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if df.empty:
    st.error("No valid dates were found in the Excel file.")
    st.stop()

# ---------------- PARAMETERS ----------------
parameters = {
    "pH": ("pH_Krishna", "pH _Panchgaga"),
    "Turbidity": ("Turbidity_Krishna", "Turbidity_Panchgaga"),
    "TDS": ("TDS_Krishna", "TDS_Panchgaga"),
    "Chloride": ("Chloride_Krishna", "Chloride_Panchgaga"),
    "Alkalinity": ("Alkalinity_Krishna", "Alkalinity_Panchgaga"),
    "Total Hardness": ("Hardness_Krishna", "Hardness_Panchgaga")
}

# ---------------- DATA OVERVIEW ----------------
st.header("1. Uploaded Data Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Readings", len(df))
c2.metric("First Reading", df["Date"].min().strftime("%d-%m-%Y"))
c3.metric("Latest Reading", df["Date"].max().strftime("%d-%m-%Y"))

with st.expander("View Uploaded Excel Data"):
    st.dataframe(df, use_container_width=True)

# ---------------- USER SELECTION ----------------
st.header("2. Select Analysis Options")

col1, col2, col3 = st.columns(3)

with col1:
    parameter = st.selectbox(
        "Select Water Quality Parameter",
        list(parameters.keys())
    )

with col2:
    river = st.selectbox(
        "Select River",
        ["Krishna", "Panchganga"]
    )

with col3:
    years = st.selectbox(
        "Future Prediction Period",
        [1, 2],
        format_func=lambda x: f"{x} Year" if x == 1 else f"{x} Years"
    )

krishna_col, panchganga_col = parameters[parameter]
selected_col = (
    krishna_col if river == "Krishna" else panchganga_col
)

if selected_col not in df.columns:
    st.error(
        f"Column '{selected_col}' is missing from your Excel file. "
        "Please check the spelling of your column names."
    )
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Convert readings to numeric values
df[selected_col] = pd.to_numeric(df[selected_col], errors="coerce")
valid_df = df[["Date", selected_col]].dropna().copy()

if valid_df.empty:
    st.error("No valid numeric readings found for this selection.")
    st.stop()

# ---------------- MONTHLY DATA ----------------
monthly = (
    valid_df.set_index("Date")[selected_col]
    .resample("MS")
    .mean()
    .dropna()
)

if len(monthly) < 6:
    st.error(
        "At least 6 months of valid readings are needed "
        "to create a prediction."
    )
    st.stop()

st.subheader(f"{parameter} — {river} River")
st.caption(
    "The graph uses monthly average values from your uploaded dataset."
)

# ---------------- ACTUAL READINGS GRAPH ----------------
st.header("3. Historical Water Quality Trend")

fig1, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(
    monthly.index,
    monthly.values,
    marker="o",
    linewidth=2,
    label="Historical Monthly Average"
)
ax1.set_xlabel("Date")
ax1.set_ylabel(parameter)
ax1.set_title(f"Historical {parameter} Readings — {river} River")
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.autofmt_xdate()
st.pyplot(fig1)
plt.close(fig1)

# ---------------- MACHINE LEARNING MODEL ----------------
st.header("4. Machine Learning Prediction")

st.write(
    "Algorithm used: **Random Forest Regressor**. "
    "The model learns from previous monthly readings "
    "to estimate future values."
)

# Create lag features from the historical monthly series
series = monthly.astype(float).copy()

training = pd.DataFrame({
    "lag1": series.shift(1),
    "lag2": series.shift(2),
    "lag3": series.shift(3),
    "target": series
}).dropna()

if len(training) < 3:
    st.error("There is not enough historical data to train the model.")
    st.stop()

X = training[["lag1", "lag2", "lag3"]]
y = training["target"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X, y)

# Predict one month at a time using the latest 3 values
future_dates = pd.date_range(
    start=series.index[-1] + pd.offsets.MonthBegin(1),
    periods=years * 12,
    freq="MS"
)

history = list(series.values)
predictions = []

for future_date in future_dates:
    features = np.array([[
        history[-1],
        history[-2],
        history[-3]
    ]])

    prediction = float(model.predict(features)[0])
    predictions.append(prediction)
    history.append(prediction)

forecast = pd.DataFrame({
    "Date": future_dates,
    "Predicted Value": predictions
})

# ---------------- ACTUAL VS FUTURE GRAPH ----------------
st.header("5. Historical and Future Prediction Graph")

fig2, ax2 = plt.subplots(figsize=(11, 5))

ax2.plot(
    series.index,
    series.values,
    marker="o",
    label="Historical Values"
)

ax2.plot(
    forecast["Date"],
    forecast["Predicted Value"],
    marker="o",
    linestyle="--",
    label="Predicted Values"
)

ax2.axvline(
    series.index[-1],
    linestyle=":",
    label="Prediction Starts"
)

ax2.set_xlabel("Date")
ax2.set_ylabel(parameter)
ax2.set_title(
    f"{parameter} — {river} River: Historical and Future Trend"
)
ax2.grid(True, alpha=0.3)
ax2.legend()
fig2.autofmt_xdate()
st.pyplot(fig2)
plt.close(fig2)

# ---------------- PREDICTION TABLE ----------------
st.header("6. Predicted Values")

forecast_display = forecast.copy()
forecast_display["Date"] = forecast_display["Date"].dt.strftime("%B %Y")
forecast_display["Predicted Value"] = (
    forecast_display["Predicted Value"].round(3)
)

st.dataframe(forecast_display, use_container_width=True)

csv = forecast.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Predictions as CSV",
    data=csv,
    file_name=f"{river}_{parameter}_predictions.csv",
    mime="text/csv"
)

st.caption(
    "Note: These are model-based estimates, not guaranteed future readings. "
    "The predictions use historical patterns only and should not be treated "
    "as proof that water is safe or unsafe to drink."
)

st.divider()
st.write("Developed for the Water Quality Monitoring project.")

