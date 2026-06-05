# ============================================================
# REGRESSIÓ SUPERVISADA - PREDICCIÓ DEL PREU AIRBNB ASHEVILLE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. CARREGAR DATASET
# ============================================================

df = pd.read_csv("dataset/listings.csv.gz", compression="gzip")

# ============================================================
# 2. NETEJAR PRICE
# ============================================================

df["price_clean"] = (
    df["price"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
)

# ============================================================
# 3. SELECCIONAR VARIABLES
# ============================================================

features = [
    "neighbourhood_cleansed",
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
    "calculated_host_listings_count_entire_homes"
]

target = "price_clean"

df_model = df[features + [target]].copy()

# Eliminem files sense preu
df_model = df_model.dropna(subset=[target])

# Eliminem preus extrems per no distorsionar el model
limit_preu = df_model[target].quantile(0.99)
df_model = df_model[df_model[target] <= limit_preu]

print("Files utilitzades pel model:", len(df_model))

# ============================================================
# 4. SEPARAR X I y
# ============================================================

X = df_model[features]
y = df_model[target]

# X = dades d'entrada
# y = preu real que volem predir

# ============================================================
# 5. SEPARAR VARIABLES CATEGÒRIQUES I NUMÈRIQUES
# ============================================================

cat_features = [
    "neighbourhood_cleansed",
    "room_type",
    "property_type"
]

num_features = [
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
    "calculated_host_listings_count_entire_homes"
]

# ============================================================
# 6. PREPROCESSAMENT
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            cat_features
        ),
        (
            "num",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_features
        )
    ]
)

# ============================================================
# 7. CREAR MODEL
# ============================================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    ))
])

# ============================================================
# 8. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dades entrenament:", len(X_train))
print("Dades test:", len(X_test))

# ============================================================
# 9. ENTRENAR MODEL
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# 10. FER PREDICCIONS
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# 11. AVALUAR RESULTATS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nRESULTATS DEL MODEL")
print("------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2:", round(r2, 3))