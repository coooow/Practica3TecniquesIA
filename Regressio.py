import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm # Per a mapes de colors avançats

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("dataset/listings.csv.gz", compression="gzip")

df["price_clean"] = (
    df["price"]
    .replace(r"[\$,]", "", regex=True)
    .astype(float)
)

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
df_model = df_model.dropna()

# Eliminem preus extrems per no distorsionar el model
limit_preu = df_model[target].quantile(0.99)
df_model = df_model[df_model[target] <= limit_preu]

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identifiquem les columnes categòriques i numèriques
categorical_cols = ["neighbourhood_cleansed", "room_type", "property_type"]
numerical_cols = [col for col in features if col not in categorical_cols]

# Preprocessing: One-Hot Encoding per a variables categòriques i mantenir les numèriques sense canvis
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ])

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R²:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

plt.figure(figsize=(10, 8))

# Dibuixem el gràfic de dispersió (Observacions Reals vs Prediccions)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.25, color="#4472C4", edgecolor=None)

# Dibuixem la recta de predicció perfecta (on el valor real és igual al predit)
max_val = y_test.max()
lims = [0, max_val]
plt.plot(lims, lims, color="#C00000", linestyle="--", linewidth=2.5, label="Predicció Perfecta (Y = X)")

# Títols orientats a conclusions
plt.title(f"El model explica el {r2*100:.1f}% de la variació dels preus", fontsize=16, loc="left", pad=15)

# Títol secundari corregit: l'RMSE ja està en dòlars reals
plt.suptitle(f"Regressió Lineal Múltiple | RMSE: {rmse:.2f} $ (marge d'error mitjà per nit)", 
             fontsize=12, color="gray", x=0.31, y=0.92)

plt.xlabel("Preu Real de l'Airbnb ($)", fontsize=12)
plt.ylabel("Preu Predit pel Model ($)", fontsize=12)
plt.xlim(lims)
plt.ylim(lims)
plt.legend(loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
