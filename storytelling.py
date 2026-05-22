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

data = pd.read_csv('AIRBNBListings.csv', encoding='latin-1', low_memory=False)  # Carrega les dades des d'un fitxer CSV
df = data.copy()  # Crea una còpia del DataFrame per a manipulacions
target = "price"  # Variable objectiu que volem explicar

df.drop(columns=["listing_id", "name", "host_id", "host_since", "host_response_time", "host_response_rate",
                  "host_acceptance_rate", "host_is_superhost", "host_total_listings_count", 
                  "host_has_profile_pic", "host_identity_verified", "district", "review_scores_rating",
                  "review_scores_accuracy", "review_scores_checkin", "review_scores_cleanliness",
                  "review_scores_location", "review_scores_value", "review_scores_communication",
                  "instant_bookable"], inplace=True)  # Elimina columnes no necessàries
df.drop(df[df["city"] != "Paris"].index, inplace=True)  # Elimina les files que no pertanyen a Paris

plt.figure(figsize=(5, 4))

# Utilitzem un scatter plot com a mapa.
# L'eix X és la longitud, eix Y latitud.
# El color (c) representa el preu (target).
# La mida (s) representa la població del districte.
scatter = plt.scatter(
    df["longitude"], 
    df["latitude"], 
    c=df[target], 
    cmap="coolwarm", # De blau (barat) a vermell (car)
    alpha=0.5, # Transparència per veure superposicions (densitat)
    edgecolor=None
)

# Afegim barra de colors explicativa
cbar = plt.colorbar(scatter)
cbar.set_label("Preu del Airbnb", rotation=270, labelpad=20)

# Títols Narratius (Storytelling)
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Espai pel suptitle
plt.show()