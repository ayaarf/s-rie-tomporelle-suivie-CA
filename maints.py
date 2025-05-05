from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
import pandas as pd
import pyodbc
import io
from prophet import Prophet
from datetime import datetime, date

app = FastAPI(title="API Transactions Monoprix")

# Connexion SQL
def get_data():
    try:
        conn = pyodbc.connect(
            'DRIVER={SQL Server};'
            'SERVER=DESKTOP-7QR5TL6;'
            'DATABASE=SA_SAP_Fico;'
            'Trusted_Connection=yes'
        )
        query = """
        SELECT TransactionDate, Amount, PaymentMethod
        FROM [dbo].[SA_Sales_Transactions]
        WHERE TransactionDate IS NOT NULL
        ORDER BY TransactionDate
        """
        df = pd.read_sql(query, conn)
        conn.close()
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        df['Amount'] = df['Amount'].astype(str).str.replace(',', '.').astype(float)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur base de données : {str(e)}")

# Endpoint 1 : Statistiques globales
@app.get("/daily-sales")
def get_daily_sales():
    df = get_data()
    daily = df.groupby('TransactionDate')['Amount'].sum().reset_index()
    return daily.to_dict(orient='records')

# Endpoint 2 : Filtrage par dates
@app.get("/daily-sales/filter")
def filter_sales(start: date, end: date):
    df = get_data()
    daily = df.groupby('TransactionDate')['Amount'].sum().reset_index()
    mask = (daily['TransactionDate'] >= pd.to_datetime(start)) & (daily['TransactionDate'] <= pd.to_datetime(end))
    result = daily[mask]
    return result.to_dict(orient='records')

# Endpoint 3 : CSV export des données filtrées
@app.get("/daily-sales/filter/download")
def download_filtered(start: date, end: date):
    df = get_data()
    daily = df.groupby('TransactionDate')['Amount'].sum().reset_index()
    mask = (daily['TransactionDate'] >= pd.to_datetime(start)) & (daily['TransactionDate'] <= pd.to_datetime(end))
    result = daily[mask]
    csv_buffer = io.StringIO()
    result.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return StreamingResponse(io.BytesIO(csv_buffer.getvalue().encode()), media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=filtered_sales.csv"
    })

# Endpoint 4 : Prédictions Prophet
@app.get("/daily-sales/predict")
def predict_sales(days: int = Query(30, ge=1, le=365)):
    df = get_data()
    daily = df.groupby('TransactionDate')['Amount'].sum().reset_index()
    prophet_df = daily.rename(columns={'TransactionDate': 'ds', 'Amount': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    result = forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'MontantPrévu',
        'yhat_lower': 'MontantPrevuMin',
        'yhat_upper': 'MontantPrevuMax'
    })[['Date', 'MontantPrevu', 'MontantPrevuMin', 'MontantPrevuMax']]
    return result.tail(days).to_dict(orient='records')
