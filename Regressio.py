import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm # Per a mapes de colors avançats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error

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
df_model = df_model.dropna(subset=[target])

# Eliminem preus extrems per no distorsionar el model
limit_preu = df_model[target].quantile(0.99)
df_model = df_model[df_model[target] <= limit_preu]

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

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
# Si el model fos perfecte, tots els punts estarien sobre aquesta línia vermella
lims = [0, 5.5]
plt.plot(lims, lims, color="#C00000", linestyle="--", linewidth=2.5, label="Predicció Perfecta (Y = X)")

# Afegim elements narratius i annotations per potenciar el Storytelling
plt.axvline(x=5.0, color="gray", linestyle=":", linewidth=1.5)
plt.text(5.05, 1.5, "Límit real de les dades (5.0)\nEl model intenta predir més enllà", 
         color="gray", fontsize=10, rotation=90)

# Títols orientats a conclusions
plt.title(f"El model explica el {r2*100:.1f}% de la variació dels preus", fontsize=16, loc="left", pad=15)
plt.suptitle(f"Regressió Lineal Múltiple | RMSE: {rmse:.3f} (error mitjà d'uns {rmse*100000:.0f} $)", 
             fontsize=12, color="gray", x=0.31, y=0.92)

plt.xlabel("Valor Real de l'Habitatge (MedHouseVal)")
plt.ylabel("Valor Predit pel Model")
plt.xlim(lims)
plt.ylim(lims)
plt.legend(loc="upper left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
