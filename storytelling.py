# IMPORTS I CONFIGURACIÓ
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm # Per a mapes de colors avançats

# Configuració estètica de Seaborn per a Storytelling
sns.set_theme(style="whitegrid", context="talk")
# Paleta de colors coherent per a tota la història
# Blau per a dades generals, Vermell per a alertes/tendències, Verd per a distribucions netes
MAIN_BLUE = "#4472C4"
ALERT_RED = "#C00000"
CLEAN_GREEN = "#70AD47"

data = pd.read_csv("dataset/listings.csv.gz", compression="gzip")  # Carrega les dades des d'un fitxer CSV
df = data.copy()  # Crea una còpia del DataFrame per a manipulacions
target = "price_clean" # Variable objectiu que volem explicar

# NETEJA DEL PREU
df["price_clean"] = (
    df["price"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
)
# COLUMNES QUE UTILITZAREM
columnes_utiles = [
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "room_type",
    "property_type",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "minimum_nights",
    "availability_90",
    "availability_365",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "review_scores_rating",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "price_clean"
]

df = df[columnes_utiles].copy()

# ELIMINAR FILES SENSE PREU O COORDENADES
df = df.dropna(subset=["price_clean", "latitude", "longitude"])

# ELIMINAR PREUS EXTREMS PER FER EL GRÀFIC MÉS LLEGIBLE
limit_preu = df["price_clean"].quantile(0.99)
df_plot = df[df["price_clean"] <= limit_preu].copy()

# COMPROVACIÓ BÀSICA
print("Files totals després de la neteja:", len(df_plot))
print("Preu mínim:", df_plot["price_clean"].min())
print("Preu màxim:", df_plot["price_clean"].max())
print("Preu mitjà:", round(df_plot["price_clean"].mean(), 2))

# MAPA DE PUNTS AMB PREU
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    df_plot["longitude"],
    df_plot["latitude"],
    c=df_plot["price_clean"],
    cmap="coolwarm",
    alpha=0.6,
    s=18
)

cbar = plt.colorbar(scatter)
cbar.set_label("Preu de l'allotjament", rotation=270, labelpad=20)


# Títols Narratius (Storytelling)
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Espai pel suptitle
plt.show()