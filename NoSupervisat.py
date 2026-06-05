# ============================================================
# APRENENTATGE NO SUPERVISAT - AGRUPACIO D'ANUNCIS AIRBNB
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# 1. CONFIGURACIO GENERAL
# ============================================================

RUTA_DATASET_GZ = "dataset/listings.csv.gz"
RUTA_DATASET_CSV = "dataset/listings.csv"
CARPETA_SORTIDA = "outputs/grafics"
CARPETA_DADES = "outputs"

os.makedirs(CARPETA_SORTIDA, exist_ok=True)
os.makedirs(CARPETA_DADES, exist_ok=True)

MAIN_BLUE = "#4472C4"
ALERT_RED = "#C00000"
CLEAN_GREEN = "#70AD47"


def crear_one_hot_encoder():
    """Compatibilitat amb versions noves i antigues de scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def convertir_percentatge(serie):
    return (
        serie.astype(str)
        .str.replace("%", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )


# ============================================================
# 2. CARREGAR DATASET
# ============================================================

if os.path.exists(RUTA_DATASET_GZ):
    df = pd.read_csv(RUTA_DATASET_GZ, compression="gzip")
elif os.path.exists(RUTA_DATASET_CSV):
    df = pd.read_csv(RUTA_DATASET_CSV)
else:
    raise FileNotFoundError("No s'ha trobat dataset/listings.csv.gz ni dataset/listings.csv")

print("Files inicials:", len(df))
print("Columnes inicials:", len(df.columns))

# ============================================================
# 3. NETEJA DE VARIABLES
# ============================================================

df["price_clean"] = (
    df["price"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
)

df["host_response_rate_clean"] = convertir_percentatge(df["host_response_rate"])
df["host_acceptance_rate_clean"] = convertir_percentatge(df["host_acceptance_rate"])
df["host_is_superhost_num"] = df["host_is_superhost"].map({"t": 1, "f": 0})
df["instant_bookable_num"] = df["instant_bookable"].map({"t": 1, "f": 0})

# Eliminem preus extrems perque no dominin la distancia dels clusters.
limit_preu_99 = df["price_clean"].quantile(0.99)
df = df[df["price_clean"] <= limit_preu_99].copy()

# ============================================================
# 4. VARIABLES PER A L'ANALISI NO SUPERVISADA
# ============================================================

# Objectiu: agrupar anuncis segons el perfil d'explotacio turistica i d'amfitrio.
num_features = [
    "price_clean",
    "accommodates",
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
    "estimated_revenue_l365d",
    "host_response_rate_clean",
    "host_acceptance_rate_clean",
    "host_listings_count",
    "host_total_listings_count",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "host_is_superhost_num",
    "instant_bookable_num",
]

cat_features = [
    "neighbourhood_cleansed",
    "room_type",
    "property_type",
]

features = num_features + cat_features
df_model = df[features].copy()

print("\nFiles utilitzades pel model no supervisat:", len(df_model))

# ============================================================
# 5. PREPROCESSAMENT
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_features
        ),
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", crear_one_hot_encoder())
            ]),
            cat_features
        )
    ]
)

X_prepared = preprocessor.fit_transform(df_model)

print("Dimensions despres del preprocessament:", X_prepared.shape)

# ============================================================
# 6. SELECCIO DEL NOMBRE DE CLUSTERS AMB KMEANS
# ============================================================

resultats_k = []

for k in range(2, 9):
    model_kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    etiquetes = model_kmeans.fit_predict(X_prepared)
    sil = silhouette_score(X_prepared, etiquetes)
    resultats_k.append({
        "k": k,
        "inertia": model_kmeans.inertia_,
        "silhouette": sil
    })

resultats_k_df = pd.DataFrame(resultats_k)
millor_k = int(resultats_k_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

print("\nRESULTATS KMEANS")
print(resultats_k_df.round(3).to_string(index=False))
print("\nMillor k segons silhouette:", millor_k)

plt.figure(figsize=(9, 5))
plt.plot(resultats_k_df["k"], resultats_k_df["silhouette"], marker="o", color=MAIN_BLUE)
plt.title("Avaluacio del nombre de clusters amb silhouette")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Silhouette score")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(f"{CARPETA_SORTIDA}/nosupervisat_silhouette_kmeans.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 5))
plt.plot(resultats_k_df["k"], resultats_k_df["inertia"], marker="o", color=ALERT_RED)
plt.title("Metode del colze per a KMeans")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inercia")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(f"{CARPETA_SORTIDA}/nosupervisat_colze_kmeans.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. MODEL FINAL KMEANS
# ============================================================

kmeans_final = KMeans(
    n_clusters=millor_k,
    random_state=42,
    n_init=20
)

df_resultat = df.copy()
df_resultat["cluster_kmeans"] = kmeans_final.fit_predict(X_prepared)

silhouette_final = silhouette_score(X_prepared, df_resultat["cluster_kmeans"])

print("\nSilhouette final KMeans:", round(silhouette_final, 3))
print("\nDistribucio de clusters:")
print(df_resultat["cluster_kmeans"].value_counts().sort_index())

# ============================================================
# 8. VISUALITZACIO DELS CLUSTERS AMB PCA
# ============================================================

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_prepared)

df_resultat["pca_1"] = X_pca[:, 0]
df_resultat["pca_2"] = X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_resultat,
    x="pca_1",
    y="pca_2",
    hue="cluster_kmeans",
    palette="tab10",
    alpha=0.75,
    s=35
)
plt.title("Clusters d'anuncis Airbnb segons PCA")
plt.xlabel("Component principal 1")
plt.ylabel("Component principal 2")
plt.legend(title="Cluster")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(f"{CARPETA_SORTIDA}/nosupervisat_clusters_pca.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 9. PERFIL I INTERPRETACIO DELS CLUSTERS
# ============================================================

perfil_numeric = (
    df_resultat
    .groupby("cluster_kmeans")[num_features]
    .mean()
    .round(2)
)

perfil_extra = (
    df_resultat
    .groupby("cluster_kmeans")
    .agg(
        anuncis=("id", "count"),
        barri_mes_frequent=("neighbourhood_cleansed", lambda x: x.mode().iloc[0]),
        tipus_habitacio_mes_frequent=("room_type", lambda x: x.mode().iloc[0]),
        tipus_propietat_mes_frequent=("property_type", lambda x: x.mode().iloc[0])
    )
)

perfil_clusters = perfil_extra.join(perfil_numeric)

print("\nPERFIL DELS CLUSTERS")
print(perfil_clusters.to_string())

perfil_clusters.to_csv(f"{CARPETA_DADES}/nosupervisat_perfil_clusters.csv")
resultats_k_df.to_csv(f"{CARPETA_DADES}/nosupervisat_resultats_kmeans.csv", index=False)
df_resultat[[
    "id",
    "name",
    "neighbourhood_cleansed",
    "room_type",
    "property_type",
    "price_clean",
    "cluster_kmeans",
    "pca_1",
    "pca_2"
]].to_csv(f"{CARPETA_DADES}/nosupervisat_anuncis_clusteritzats.csv", index=False)

plt.figure(figsize=(11, 6))
sns.heatmap(
    perfil_numeric[[
        "price_clean",
        "availability_365",
        "number_of_reviews",
        "estimated_occupancy_l365d",
        "estimated_revenue_l365d",
        "host_listings_count",
        "calculated_host_listings_count_entire_homes",
        "host_is_superhost_num"
    ]],
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)
plt.title("Perfil numeric mitja de cada cluster")
plt.xlabel("Variable")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(f"{CARPETA_SORTIDA}/nosupervisat_perfil_clusters_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 10. COMPARACIO AMB DBSCAN
# ============================================================

resultats_dbscan = []

for eps in [1.5, 2.0, 2.5, 3.0, 3.5]:
    dbscan = DBSCAN(eps=eps, min_samples=10)
    etiquetes_db = dbscan.fit_predict(X_prepared)

    n_clusters = len(set(etiquetes_db)) - (1 if -1 in etiquetes_db else 0)
    soroll = int(np.sum(etiquetes_db == -1))

    if n_clusters >= 2:
        mascara_valids = etiquetes_db != -1
        sil_db = silhouette_score(X_prepared[mascara_valids], etiquetes_db[mascara_valids])
    else:
        sil_db = np.nan

    resultats_dbscan.append({
        "eps": eps,
        "clusters": n_clusters,
        "soroll": soroll,
        "silhouette_sense_soroll": sil_db
    })

resultats_dbscan_df = pd.DataFrame(resultats_dbscan)

print("\nRESULTATS DBSCAN")
print(resultats_dbscan_df.round(3).to_string(index=False))

resultats_dbscan_df.to_csv(f"{CARPETA_DADES}/nosupervisat_resultats_dbscan.csv", index=False)

