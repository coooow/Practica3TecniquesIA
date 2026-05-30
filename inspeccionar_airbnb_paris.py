import pandas as pd

df = pd.read_csv("listings.csv.gz", compression="gzip")

print("Columnes relacionades amb preu:")
for col in df.columns:
    if "price" in col.lower() or "rate" in col.lower() or "cost" in col.lower():
        print(col)
        print("Valors no nuls:", df[col].notna().sum())
        print(df[col].dropna().head(10))
        print("-" * 40)