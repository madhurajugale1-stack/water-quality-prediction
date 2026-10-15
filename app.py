
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Water Quality Monitoring",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Water Quality Monitoring & Prediction")
st.write("Krishna and Panchganga River Analysis")

uploaded_file = st.file_uploader(
    "Upload your Water Quality Excel file",
    type=["xlsx", "xls"]
)

parameters = {
    "pH": ["pH_Krishna", "pH _Panchgaga"],
    "Turbidity": ["Turbidity_Krishna", "Turbidity_Panchgaga"],
    "TDS": ["TDS_Krishna", "TDS_Panchgaga"],
    "Chloride": ["Chloride_Krishna", "Chloride_Panchgaga"],
    "Alkalinity": ["Alkalinity_Krishna", "Alkalinity_Panchgaga"],
    "Hardness": ["Hardness_Krishna", "Hardness_Panchgaga"]
}

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    if "Date" not in df.columns:
        st.error("Your Excel file must contain a Date column.")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    parameter = st.selectbox(
        "Select Water Quality Parameter",
        list(parameters.keys())
    )

    river = st.selectbox(
        "Select River",
        ["Both Rivers", "Krishna", "Panchganga"]
    )

    years = st.selectbox(
        "Prediction Period",
        [1, 2],
        index=1,
        format_func=lambda x: f"{x} Year{'s' if x > 1 else ''}"
    )

    selected_columns = parameters[parameter]

    river_columns = []
    if river in ["Both Rivers", "Krishna"]:
        river_columns.append((selected_columns[0], "Krishna"))
    if river in ["Both Rivers", "Panchganga"]:
        river_columns.append((selected_columns[1], "Panchganga"))

    st.subheader(f"{parameter}: Actual Readings")

    fig, ax = plt.subplots(figsize=(12, 5))

    for column, river_name in river_columns:
        if column not in df.columns:
            st.warning(f"Column not found: {column}")
            continue

        ax.plot(
            df["Date"], pd.to_numeric(df[column], errors="coerce"),
            marker="o", markersize=3, label=f"{river_name} Actual"
        )

    ax.set_xlabel("Date")
    ax.set_ylabel(parameter)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader(f"{parameter}: Future Prediction")

    forecast_fig, forecast_ax = plt.subplots(figsize=(12, 5))
    prediction_tables = []

    for column, river_name in river_columns:
        if column not in df.columns:
            continue

        series = df.set_index("Date")[column]
        series = pd.to_numeric(series, errors="coerce").dropna()

        monthly = series.resample("MS").mean()
        monthly = monthly.interpolate(method="time").ffill().bfill()
        monthly = monthly.dropna()

        if len(monthly) < 6:
            st.warning(
                f"Not enough monthly readings for {river_name}."
            )
            continue

        values = monthly.to_numpy(dtype=float)

        training = pd.DataFrame({
            "lag1": values[2:-1],
            "lag2": values[1:-2],
            "lag3": values[:-3],
            "target": values[3:]
        })

        if training.empty:
            continue

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        model.fit(
            training[["lag1", "lag2", "lag3"]],
            training["target"]
        )

        history = list(values)
        predictions = []

        for _ in range(12 * years):
            features = [[history[-1], history[-2], history[-3]]]
            predicted_value = float(model.predict(features)[0])
            predictions.append(predicted_value)
            history.append(predicted_value)

        future_dates = pd.date_range(
            start=monthly.index[-1] + pd.offsets.MonthBegin(1),
            periods=12 * years,
            freq="MS"
        )

        forecast_ax.plot(
            future_dates,
            predictions,
            marker="o",
            markersize=3,
            label=f"{river_name} Predicted"
        )

        prediction_tables.append(
            pd.DataFrame({
                "Date": future_dates,
                "River": river_name,
                "Parameter": parameter,
                "Predicted Value": predictions
            })
        )

    forecast_ax.set_xlabel("Date")
    forecast_ax.set_ylabel(parameter)
    forecast_ax.grid(True)
    forecast_ax.legend()
    forecast_fig.tight_layout()
    st.pyplot(forecast_fig)
    plt.close(forecast_fig)

    if prediction_tables:
        prediction_df = pd.concat(
            prediction_tables, ignore_index=True
        )

        st.subheader("Predicted Values")
        st.dataframe(prediction_df, use_container_width=True)

        st.download_button(
            "Download Predictions as CSV",
            prediction_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{parameter.lower()}_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            "These are model-based estimates, not guaranteed future "
            "measurements. Validate model accuracy before using them "
            "for water safety decisions."
        )

else:
    st.info("Upload your Excel file to view graphs and predictions.")
