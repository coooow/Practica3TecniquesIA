import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm # Per a mapes de colors avançats

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

CARPETA_SORTIDA = "outputs/grafics"

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

df = df_model.dropna()

bins = [0, 50, 150, 300, np.inf]
labels = ['0-50', '50-150', '150-300', '300+']

df['price_category'] = pd.cut(df['price_clean'], bins=bins, labels=labels)

print("Distribució de les categories de preu:")
print(df['price_category'].value_counts())

X = df.drop(columns=['price_clean', 'price_category'])
y = df['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

model_arbre_cat = DecisionTreeClassifier(random_state=42, max_depth=3, class_weight='balanced')
model_arbre_cat.fit(X_train_encoded, y_train)

y_pred_cat = model_arbre_cat.predict(X_test_encoded)

print("Report de classificació:")
print(classification_report(y_test, y_pred_cat))

cm = confusion_matrix(y_test, y_pred_cat)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Matriu de Confusió: On s'equivoca l'arbre?", fontsize=16, pad=15)
plt.xlabel("Predicció del Model")
plt.ylabel("Realitat (Segment Real)")
plt.tight_layout()

plt.savefig(f"{CARPETA_SORTIDA}/matriu_confusio.png", dpi=300, bbox_inches="tight")

plt.show()

# DIBUIX DE L'ARBRE DE CLASSIFICACIÓ
plt.figure(figsize=(18, 8))
plot_tree(
    model_arbre_cat,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=labels,
    filled=True,
    rounded=True,
    fontsize=11,
    precision=2
)

plt.title("Arbre de Classificació per a la segmentació del preu de l'Airbnb", fontsize=18, loc="left", pad=20)
plt.tight_layout()

plt.savefig(f"{CARPETA_SORTIDA}/arbre_classificacio.png", dpi=300, bbox_inches="tight")

plt.show()