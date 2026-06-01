# ============================================================
# PRÀCTICA 3 - AIRBNB ASHEVILLE
# EXPLORACIÓ INICIAL I GRÀFICS PER AL STORYTELLING
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ============================================================
# 1. CONFIGURACIÓ GENERAL
# ============================================================

RUTA_DATASET = "dataset/listings.csv.gz"
CARPETA_SORTIDA = "outputs/grafics"

os.makedirs(CARPETA_SORTIDA, exist_ok=True)

# Colors principals
MAIN_BLUE = "#4472C4"
ALERT_RED = "#C00000"
CLEAN_GREEN = "#70AD47"

# ============================================================
# 2. CARREGAR DATASET
# ============================================================

data = pd.read_csv(RUTA_DATASET, compression="gzip")
df = data.copy()

print("Files inicials:", len(df))
print("Columnes inicials:", len(df.columns))

# ============================================================
# 3. NETEJA DE LA VARIABLE PRICE
# ============================================================

# Airbnb acostuma a guardar el preu com a text: "$120.00"
df["price_clean"] = (
    df["price"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# ============================================================
# 4. SELECCIÓ DE COLUMNES ÚTILS
# ============================================================

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
    "estimated_occupancy_l365d",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "price_clean"
]

df = df[columnes_utiles].copy()

# ============================================================
# 5. ELIMINACIÓ DE NULS BÀSICS
# ============================================================

df = df.dropna(subset=[
    "price_clean",
    "latitude",
    "longitude",
    "neighbourhood_cleansed"
])

print("\nFiles després d'eliminar nuls bàsics:", len(df))

# ============================================================
# 6. ELIMINAR PREUS EXTREMS PER FER GRÀFICS MÉS LLEGIBLES
# ============================================================

# Per visualització, eliminem el 1% més car.
# Això evita que uns pocs allotjaments molt cars deformin l'escala de color.
limit_preu_99 = df["price_clean"].quantile(0.99)

df_plot = df[df["price_clean"] <= limit_preu_99].copy()

print("\nFiles totals després de la neteja:", len(df_plot))
print("Preu mínim:", df_plot["price_clean"].min())
print("Preu màxim:", df_plot["price_clean"].max())
print("Preu mitjà:", round(df_plot["price_clean"].mean(), 2))
print("Mediana del preu:", round(df_plot["price_clean"].median(), 2))

# ============================================================
# 7. MAPA MILLORAT DE PREUS
# ============================================================

plt.figure(figsize=(10, 7))

# Limitem l'escala visual al percentil 95 perquè el color sigui més informatiu
limit_color_95 = df_plot["price_clean"].quantile(0.95)

norm = Normalize(
    vmin=df_plot["price_clean"].min(),
    vmax=limit_color_95
)

scatter = plt.scatter(
    df_plot["longitude"],
    df_plot["latitude"],
    c=df_plot["price_clean"],
    cmap="coolwarm",
    norm=norm,
    alpha=0.65,
    s=22,
    edgecolors="none"
)

cbar = plt.colorbar(scatter)
cbar.set_label("Preu de l'allotjament ($)", rotation=270, labelpad=22)

plt.title("Distribució geogràfica dels preus d'Airbnb a Asheville", fontsize=15)
plt.xlabel("Longitud")
plt.ylabel("Latitud")

plt.grid(True, alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{CARPETA_SORTIDA}/mapa_preus_asheville_millorat.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 8. GRÀFIC DE PREU MITJÀ PER BARRI
# ============================================================

preu_barri = (
    df_plot
    .groupby("neighbourhood_cleansed")["price_clean"]
    .agg(["count", "mean", "median"])
    .sort_values(by="mean", ascending=False)
)

# Ens quedem amb barris amb un mínim d'anuncis per evitar conclusions febles
preu_barri_filtrat = preu_barri[preu_barri["count"] >= 10].head(15)

plt.figure(figsize=(11, 7))

plt.barh(
    preu_barri_filtrat.index[::-1],
    preu_barri_filtrat["mean"][::-1],
    color=MAIN_BLUE
)

plt.title("Barris amb preu mitjà més alt a Asheville", fontsize=15)
plt.xlabel("Preu mitjà ($)")
plt.ylabel("Barri")

plt.grid(axis="x", alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{CARPETA_SORTIDA}/preu_mitja_per_barri_asheville.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 9. DISTRIBUCIÓ DEL PREU
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df_plot["price_clean"],
    bins=40,
    color=MAIN_BLUE,
    alpha=0.8
)

plt.axvline(
    df_plot["price_clean"].mean(),
    color=ALERT_RED,
    linestyle="--",
    linewidth=2,
    label=f"Mitjana: {round(df_plot['price_clean'].mean(), 2)} $"
)

plt.axvline(
    df_plot["price_clean"].median(),
    color=CLEAN_GREEN,
    linestyle="--",
    linewidth=2,
    label=f"Mediana: {round(df_plot['price_clean'].median(), 2)} $"
)

plt.title("Distribució dels preus dels allotjaments", fontsize=15)
plt.xlabel("Preu ($)")
plt.ylabel("Nombre d'allotjaments")
plt.legend()

plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{CARPETA_SORTIDA}/distribucio_preus_asheville.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 10. TIPUS D'ALLOTJAMENT
# ============================================================

room_type_counts = df_plot["room_type"].value_counts()

plt.figure(figsize=(9, 6))

plt.bar(
    room_type_counts.index,
    room_type_counts.values,
    color=MAIN_BLUE
)

plt.title("Tipus d'allotjament a Airbnb Asheville", fontsize=15)
plt.xlabel("Tipus d'allotjament")
plt.ylabel("Nombre d'anuncis")

plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

plt.savefig(
    f"{CARPETA_SORTIDA}/tipus_allotjament_asheville.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nRESUM FINAL")
print("----------")
print("Anuncis inicials:", len(data))
print("Anuncis amb preu i coordenades vàlides:", len(df))
print("Anuncis utilitzats als gràfics:", len(df_plot))
print("Preu mínim:", df_plot["price_clean"].min())
print("Preu màxim visualitzat:", df_plot["price_clean"].max())
print("Preu mitjà:", round(df_plot["price_clean"].mean(), 2))
print("Mediana:", round(df_plot["price_clean"].median(), 2))

print("\nGràfics guardats a:")
print(CARPETA_SORTIDA)

plt.show()

# ============================================================
# 11. RESUM FINAL
# ============================================================

