import pandas as pd
from pathlib import Path


# ==============================
# CONFIGURACIÓ
# ==============================

FITXER_CSV = "dataset/calender.csv"
CARPETA_SORTIDA = "sortida_columnes"

Path(CARPETA_SORTIDA).mkdir(exist_ok=True)


# ==============================
# LECTURA DEL CSV
# ==============================

try:
    df = pd.read_csv(FITXER_CSV)
except FileNotFoundError:
    print(f"No s'ha trobat el fitxer: {FITXER_CSV}")
    print("Comprova que la carpeta 'dataset' existeixi i que el fitxer es digui exactament 'calender.csv'.")
    exit()
except Exception as e:
    print("Error llegint el CSV:")
    print(e)
    exit()


# ==============================
# INFORMACIÓ GENERAL
# ==============================

print("\n====================================")
print(" INFORMACIÓ GENERAL DEL DATASET")
print("====================================")
print(f"Fitxer: {FITXER_CSV}")
print(f"Files: {df.shape[0]}")
print(f"Columnes: {df.shape[1]}")

print("\nColumnes trobades:")
for i, columna in enumerate(df.columns, start=1):
    print(f"{i}. {columna}")


# ==============================
# RESUM PER COLUMNES
# ==============================

resum_columnes = []

for columna in df.columns:
    tipus = df[columna].dtype
    nuls = df[columna].isnull().sum()
    percentatge_nuls = round((nuls / len(df)) * 100, 2)
    valors_unics = df[columna].nunique()

    exemples = (
        df[columna]
        .dropna()
        .astype(str)
        .unique()[:3]
    )

    exemples_text = " | ".join(exemples)

    resum_columnes.append({
        "columna": columna,
        "tipus_dada": tipus,
        "nuls": nuls,
        "percentatge_nuls": percentatge_nuls,
        "valors_unics": valors_unics,
        "exemples": exemples_text
    })

df_resum = pd.DataFrame(resum_columnes)


# ==============================
# MOSTRAR RESUM EN TERMINAL
# ==============================

print("\n====================================")
print(" RESUM DE COLUMNES")
print("====================================")
print(df_resum.to_string(index=False))


# ==============================
# GUARDAR RESUM EN CSV
# ==============================

ruta_csv = Path(CARPETA_SORTIDA) / "resum_columnes_calender.csv"
df_resum.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

print(f"\nResum guardat en CSV: {ruta_csv}")


# ==============================
# GUARDAR RESUM EN HTML
# ==============================

html = df_resum.to_html(index=False, border=0)

html_complet = f"""
<!DOCTYPE html>
<html lang="ca">
<head>
    <meta charset="UTF-8">
    <title>Resum de columnes - Airbnb París</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f4f4f4;
        }}

        h1 {{
            color: #222;
        }}

        .info {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
        }}

        th {{
            background-color: #222;
            color: white;
            padding: 10px;
            text-align: left;
        }}

        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }}

        tr:hover {{
            background-color: #f1f1f1;
        }}
    </style>
</head>
<body>

    <h1>Resum de columnes - Airbnb París</h1>

    <div class="info">
        <p><strong>Fitxer analitzat:</strong> {FITXER_CSV}</p>
        <p><strong>Nombre de files:</strong> {df.shape[0]}</p>
        <p><strong>Nombre de columnes:</strong> {df.shape[1]}</p>
    </div>

    {html}

</body>
</html>
"""

ruta_html = Path(CARPETA_SORTIDA) / "resum_columnes_calender.html"

with open(ruta_html, "w", encoding="utf-8") as f:
    f.write(html_complet)

print(f"Resum guardat en HTML: {ruta_html}")


# ==============================
# PRIMERES FILES
# ==============================

ruta_preview = Path(CARPETA_SORTIDA) / "primeres_files_calender.csv"
df.head(20).to_csv(ruta_preview, index=False, encoding="utf-8-sig")

print(f"Primeres 20 files guardades en: {ruta_preview}")


# ==============================
# INFORME DETALLAT
# ==============================

ruta_txt = Path(CARPETA_SORTIDA) / "detall_columnes_calender.txt"

with open(ruta_txt, "w", encoding="utf-8") as f:
    f.write("DETALL DE COLUMNES - AIRBNB PARÍS\n")
    f.write("=" * 40 + "\n\n")

    for columna in df.columns:
        f.write(f"COLUMNA: {columna}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tipus de dada: {df[columna].dtype}\n")
        f.write(f"Valors nuls: {df[columna].isnull().sum()}\n")
        f.write(f"Valors únics: {df[columna].nunique()}\n")

        f.write("Exemples:\n")
        exemples = df[columna].dropna().astype(str).unique()[:10]

        for exemple in exemples:
            f.write(f"  - {exemple}\n")

        f.write("\n\n")

print(f"Informe detallat guardat en TXT: {ruta_txt}")

print("\nProcés finalitzat correctament.")