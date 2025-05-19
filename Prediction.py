import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

sns.set(style="whitegrid")
st.set_page_config(layout="wide")
st.title("📊 Unified Predictive Modeling App")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

with st.expander("ℹ️ Instructions"):
    st.markdown("""
    - Upload a CSV dataset.
    - Choose your **modeling approach**: ARIMA (Time Series) or ML models (Random Forest, Gradient Boosting, KNN).
    - If ML model: Choose target + features.
    - If ARIMA: Choose date and target column.
    """)

def auto_detect_columns(df):
    date_col, target_col = None, None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col])
                date_col = col
                break
            except:
                continue
    if date_col is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [col for col in numeric_cols if col != date_col]
    keyword_priority = ['caseload', 'cases', 'value', 'count', 'target']
    for kw in keyword_priority:
        for col in candidates:
            if kw in col.lower():
                target_col = col
                break
        if target_col:
            break
    if target_col is None and candidates:
        target_col = df[candidates].var().idxmax()
    return date_col, target_col

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(data.head())

    model_choice = st.selectbox("🔧 Choose a modeling approach", ["Random Forest", "Gradient Boosting", "KNN", "ARIMA"])

    if model_choice == "ARIMA":
        auto_date_col, auto_target_col = auto_detect_columns(data)
        date_col = st.selectbox("Select Date Column", data.columns, index=data.columns.get_loc(auto_date_col) if auto_date_col else 0)
        target_col = st.selectbox("Select Target Column", data.columns, index=data.columns.get_loc(auto_target_col) if auto_target_col else 1)

        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data = data.dropna(subset=[date_col])
        data = data.sort_values(by=date_col)
        data[target_col] = data[target_col].ffill()

        st.subheader("📈 Time Series Plot")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(data=data, x=date_col, y=target_col, ax=ax, color='dodgerblue')
        ax.set_title('Target Variable Over Time')
        st.pyplot(fig)

        train_size = st.slider("Training Size (%)", 50, 95, 80)
        train_data, test_data = train_test_split(data, test_size=(100-train_size)/100, shuffle=False)

        st.subheader("🔧 ARIMA Configuration")
        p = st.number_input("AR (p)", min_value=0, value=5)
        d = st.number_input("I (d)", min_value=0, value=1)
        q = st.number_input("MA (q)", min_value=0, value=0)

        try:
            model = ARIMA(train_data[target_col], order=(p, d, q))
            arima_model = model.fit()
            st.success("Model trained successfully.")

            with st.expander("ARIMA Summary"):
                st.text(arima_model.summary())

            start, end = len(train_data), len(train_data) + len(test_data) - 1
            predictions = arima_model.predict(start=start, end=end, dynamic=False)

            st.subheader("📏 Evaluation Metrics")
            mae = mean_absolute_error(test_data[target_col], predictions)
            mse = mean_squared_error(test_data[target_col], predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(test_data[target_col], predictions)

            st.metric("MAE", f"{mae:.2f}")
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("R²", f"{r2:.2f}")

            st.subheader("📊 Actual vs Predicted")
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            sns.lineplot(x=test_data[date_col], y=test_data[target_col], label='Actual', ax=ax2)
            sns.lineplot(x=test_data[date_col], y=predictions, label='Predicted', ax=ax2)
            st.pyplot(fig2)

            st.subheader("🔮 Forecast Future")
            steps = st.slider("Forecast Steps (Days)", 7, 90, 30)
            forecast = arima_model.forecast(steps=steps)
            last_date = data[date_col].iloc[-1]
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]
            future_df = pd.DataFrame({date_col: future_dates, 'Forecast': forecast})

            fig3, ax3 = plt.subplots(figsize=(14, 6))
            sns.lineplot(data=data, x=date_col, y=target_col, ax=ax3, label='Historical')
            sns.lineplot(data=future_df, x=date_col, y='Forecast', ax=ax3, label='Forecast')
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"ARIMA error: {e}")

    else:
        target = st.selectbox("🎯 Select Target Column", options=data.columns)
        features = st.multiselect("🔍 Select Feature Columns", options=[col for col in data.columns if col != target])
        
        if st.button("🚀 Run Prediction"):
            X = data[features]
            y = data[target]

            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
                                             remainder='passthrough')

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_enc = preprocessor.fit_transform(X_train)
            X_test_enc = preprocessor.transform(X_test)

            is_classification = y.dtype == 'object' or len(y.unique()) < 10

            if is_classification:
                st.info("Detected Classification Problem.")
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)
                    y_test = le.transform(y_test)

                model = RandomForestClassifier(random_state=42)
                model.fit(X_train_enc, y_train)
                preds = model.predict(X_test_enc)
                acc = accuracy_score(y_test, preds)
                st.success(f"✅ Accuracy: {acc:.2f}")
            else:
                st.info("Detected Regression Problem.")
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(random_state=42)
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor(random_state=42)
                elif model_choice == "KNN":
                    model = KNeighborsRegressor()

                model.fit(X_train_enc, y_train)
                preds = model.predict(X_test_enc)

                mse = mean_squared_error(y_test, preds)
                st.success(f"✅ MSE: {mse:.2f}")

                st.write("📉 Actual vs Predicted")
                plt.figure(figsize=(6, 5))
                plt.scatter(y_test, preds, alpha=0.7)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                st.pyplot(plt)

                st.write("📊 Residual Distribution")
                residuals = y_test - preds
                plt.figure(figsize=(6, 4))
                sns.histplot(residuals, kde=True)
                st.pyplot(plt)

                st.write("🔍 Feature Importances")
                try:
                    importances = model.feature_importances_
                    feat_names = preprocessor.get_feature_names_out()
                    feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
                    st.bar_chart(feat_imp)
                except AttributeError:
                    st.warning("Feature importances not available for this model.")