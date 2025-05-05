import streamlit as st
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from prophet import Prophet

# App title
st.title("Daily Transaction Analysis - Monoprix France")

# Connect to SQL Server database
try:
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=DESKTOP-7QR5TL6;'
        'DATABASE=SA_SAP_Fico;'
        'Trusted_Connection=yes'
    )
    st.success("Successfully connected to the database")
except pyodbc.Error as e:
    st.error(f"Database connection error: {e}")
    st.stop()

# SQL query
query = """
SELECT
    [TransactionDate],
    [Amount],
    [PaymentMethod]
FROM [dbo].[SA_Sales_Transactions]
WHERE [TransactionDate] IS NOT NULL
ORDER BY [TransactionDate]
"""

# Load data
try:
    df = pd.read_sql(query, conn)
    st.success(f"{len(df)} transactions successfully loaded")
except Exception as e:
    st.error(f"Error loading data: {e}")
    conn.close()
    st.stop()

# Close the connection
conn.close()

# Data preprocessing
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['Amount'] = df['Amount'].astype(str).str.replace(',', '.').astype(float)
df = df.drop(columns='PaymentMethod')

# Group by date
daily_sales = df.groupby(['TransactionDate'])['Amount'].sum().reset_index()

# Show data preview
st.subheader("Data Preview")
st.write(daily_sales.head())

# Descriptive statistics
st.subheader("Descriptive Statistics")
st.write(daily_sales.describe())

# Daily sales visualization
st.subheader("Amount Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='TransactionDate', y='Amount', data=daily_sales, ax=ax)
ax.set_title('Daily Sales Amount')
ax.set_xlabel('Date')
ax.set_ylabel('Amount (€)')
plt.xticks(rotation=45)
st.pyplot(fig)

# Date range filtering
st.subheader("Filter by Date Range")
min_date = daily_sales['TransactionDate'].min()
max_date = daily_sales['TransactionDate'].max()

start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

# Convert selected dates to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter data
filtered_data = daily_sales[(daily_sales['TransactionDate'] >= start_date) & (daily_sales['TransactionDate'] <= end_date)]

# Show filtered data
st.subheader("Filtered Data")
st.write(filtered_data)

# Download filtered data
st.subheader("Download Filtered Data")
if not filtered_data.empty:
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="filtered_daily_sales.csv",
        mime="text/csv"
    )

# Forecast section with Prophet
st.subheader("Amount Forecast")

# Prepare data for Prophet
prophet_df = daily_sales.rename(columns={'TransactionDate': 'ds', 'Amount': 'y'})

# Input for number of forecast days
forecast_days = st.number_input("Number of days to forecast", min_value=1, max_value=365, value=30)

# Button to trigger forecast
if st.button("Run Forecast"):
    # Initialize and train the Prophet model
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)

    # Create future dates DataFrame
    future = model.make_future_dataframe(periods=forecast_days)

    # Make predictions
    forecast = model.predict(future)

    # Rename columns for clarity
    forecast = forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'PredictedAmount',
        'yhat_lower': 'PredictedMin',
        'yhat_upper': 'PredictedMax'
    })

    # Show forecast
    st.subheader("Forecast Results")
    st.write(forecast[['Date', 'PredictedAmount', 'PredictedMin', 'PredictedMax']].tail(forecast_days))

    # Download forecast
    st.subheader("Download Forecast")
    forecast_csv = forecast[['Date', 'PredictedAmount', 'PredictedMin', 'PredictedMax']].to_csv(index=False)
    st.download_button(
        label="Download Forecast as CSV",
        data=forecast_csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )
