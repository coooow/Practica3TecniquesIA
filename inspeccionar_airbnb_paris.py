import pandas as pd

df = pd.read_csv("listings.csv.gz", compression="gzip")

columnas_interes = [
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
    "calculated_host_listings_count_entire_homes"
]

columnas_existentes = [col for col in columnas_interes if col in df.columns]
columnas_no_encontradas = [col for col in columnas_interes if col not in df.columns]

print("Columnas encontradas:")
print(columnas_existentes)

print("\nColumnas NO encontradas:")
print(columnas_no_encontradas)

resumen_nulls = pd.DataFrame({
    "columna": columnas_existentes,
    "valores_totales": [len(df)] * len(columnas_existentes),
    "valores_no_nulos": [df[col].notnull().sum() for col in columnas_existentes],
    "valores_nulos": [df[col].isnull().sum() for col in columnas_existentes],
    "porcentaje_nulos": [round(df[col].isnull().mean() * 100, 2) for col in columnas_existentes]
})

resumen_nulls["estado"] = resumen_nulls["porcentaje_nulos"].apply(
    lambda x: "Vacía o casi vacía" if x >= 95 else
              "Muchos nulos" if x >= 50 else
              "Apta"
)

resumen_nulls = resumen_nulls.sort_values(by="porcentaje_nulos", ascending=False)

print("\nResumen de nulos:")
print(resumen_nulls.to_string(index=False))