import streamlit as st
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from prophet import Prophet

# Titre de l'application
st.title("Analyse des Transactions Journalières - Monoprix France")

# Connexion à la base de données SQL Server
try:
    conn = pyodbc.connect(
        'DRIVER={SQL Server};'
        'SERVER=DESKTOP-7QR5TL6;'
        'DATABASE=SA_SAP_Fico;'
        'Trusted_Connection=yes'
    )
    st.success("Connexion à la base de données établie avec succès")
except pyodbc.Error as e:
    st.error(f"Erreur de connexion à la base de données : {e}")
    st.stop()

# Requête SQL
query = """
SELECT
    [TransactionDate],
    [Amount],
    [PaymentMethod]
FROM [dbo].[SA_Sales_Transactions]
WHERE [TransactionDate] IS NOT NULL
ORDER BY [TransactionDate]
"""

# Charger les données
try:
    df = pd.read_sql(query, conn)
    st.success(f"{len(df)} transactions chargées avec succès")
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")
    conn.close()
    st.stop()

# Fermer la connexion
conn.close()

# Prétraitement des données
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
df['Amount'] = df['Amount'].astype(str).str.replace(',', '.').astype(float)
df = df.drop(columns='PaymentMethod')

# Grouper par date 
daily_sales = df.groupby(['TransactionDate'])['Amount'].sum().reset_index()

# Afficher un aperçu des données
st.subheader("Aperçu des données")
st.write(daily_sales.head())

# Statistiques descriptives
st.subheader("Statistiques descriptives")
st.write(daily_sales.describe())

# Visualisation des ventes journalières
st.subheader("Visualisation des ventes journalières")
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(x='TransactionDate', y='Amount', data=daily_sales, ax=ax)
ax.set_title('Montant des ventes journalières')
ax.set_xlabel('Date')
ax.set_ylabel('Montant (€)')
plt.xticks(rotation=45)
st.pyplot(fig)

# Filtrer par plage de dates
st.subheader("Filtrer par plage de dates")
min_date = daily_sales['TransactionDate'].min()
max_date = daily_sales['TransactionDate'].max()

start_date = st.date_input("Date de début", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("Date de fin", min_value=min_date, max_value=max_date, value=max_date)

# Convertir les dates sélectionnées en datetime pour le filtrage
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filtrer les données
filtered_data = daily_sales[(daily_sales['TransactionDate'] >= start_date) & (daily_sales['TransactionDate'] <= end_date)]

# Afficher les données filtrées
st.subheader("Données filtrées")
st.write(filtered_data)

# Téléchargement des données filtrées
st.subheader("Télécharger les données filtrées")
if not filtered_data.empty:
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="Télécharger en CSV",
        data=csv,
        file_name="filtered_daily_sales.csv",
        mime="text/csv"
    )

# Section pour la prédiction avec Prophet
st.subheader("Prédiction des ventes journalières")

# Préparer les données pour Prophet
prophet_df = daily_sales.rename(columns={'TransactionDate': 'ds', 'Amount': 'y'})

# Input pour le nombre de jours à prévoir
forecast_days = st.number_input("Nombre de jours à prévoir", min_value=1, max_value=365, value=30)

# Bouton pour lancer la prédiction
if st.button("Lancer la prédiction"):
    # Initialiser et entraîner le modèle Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)

    # Créer un DataFrame pour les dates futures
    future = model.make_future_dataframe(periods=forecast_days)

    # Faire les prédictions
    forecast = model.predict(future)

    # Renommer les colonnes pour plus de clarté
    forecast = forecast.rename(columns={
        'ds': 'Date',
        'yhat': 'MontantPrévu',
        'yhat_lower': 'MontantPrévuMin',
        'yhat_upper': 'MontantPrévuMax'
    })

    # Afficher les prévisions
    st.subheader("Prévisions")
    st.write(forecast[['Date', 'MontantPrévu', 'MontantPrévuMin', 'MontantPrévuMax']].tail(forecast_days))

    # Téléchargement des prévisions
    st.subheader("Télécharger les prévisions")
    forecast_csv = forecast[['Date', 'MontantPrévu', 'MontantPrévuMin', 'MontantPrévuMax']].to_csv(index=False)
    st.download_button(
        label="Télécharger les prévisions en CSV",
        data=forecast_csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )