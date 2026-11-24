import pandas as pd
import requests

# 1. Parâmetros para Sceaux (arredores de Paris) e intervalo pedido
params = {
    "latitude": 48.778,
    "longitude": 2.290,
    "start_date": "2008-05-01",
    "end_date": "2009-11-01",
    "hourly": "temperature_2m",
    "timezone": "Europe/Paris",  # Mantém a hora local de França (com DST)
}

url = "https://archive-api.open-meteo.com/v1/archive"
response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
data = response.json()

# 2. Construir DataFrame horário de temperatura
df_temp = pd.DataFrame({
    "timestamp": pd.to_datetime(data["hourly"]["time"]),
    "temp_c": data["hourly"]["temperature_2m"],
})

# Guardar temperatura horária num CSV
df_temp.to_csv("data/sceaux_temp_horaria_2008_2009.csv", index=False)
print(f"Registos de temperatura obtidos: {len(df_temp)}")
print(df_temp.head())