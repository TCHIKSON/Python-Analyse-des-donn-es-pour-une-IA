# === 1. Import des librairies ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# === 2. Chargement du jeu de données ===
boston = fetch_openml(name="boston", version=1, as_frame=True)
X = boston.data
y = boston.target

# === 3. Normalisation des données ===
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# === 4. Sélection des caractéristiques ===
# Modèle simple : RM (nombre moyen de pièces par logement)
X_simple = X_scaled[["RM"]]

# Modèle multiple : RM et LSTAT (% population de statut inférieur)
X_multiple = X_scaled[["RM", "LSTAT"]]

# === 5. Division en ensembles d'entraînement et de test ===
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# === 6. Régression linéaire simple ===
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

# === 7. Régression linéaire multiple ===
model_multiple = LinearRegression()
model_multiple.fit(X_train_m, y_train_m)
y_pred_m = model_multiple.predict(X_test_m)

# === 8. Évaluation des performances ===
# Coefficient de détermination (R²)
r2_simple = r2_score(y_test_s, y_pred_s)
r2_multiple = r2_score(y_test_m, y_pred_m)

# Erreur quadratique moyenne (RMSE)
rmse_simple = np.sqrt(mean_squared_error(y_test_s, y_pred_s))
rmse_multiple = np.sqrt(mean_squared_error(y_test_m, y_pred_m))

print(" Performances des modèles :")
print(f"--- Régression simple (RM) ---")
print(f"R² = {r2_simple:.3f}")
print(f"RMSE = {rmse_simple:.3f}")

print(f"\n--- Régression multiple (RM + LSTAT) ---")
print(f"R² = {r2_multiple:.3f}")
print(f"RMSE = {rmse_multiple:.3f}")

# === 9. Visualisations ===

# (a) Relation entre RM et MEDV
plt.figure(figsize=(7,5))
sns.scatterplot(x=X["RM"], y=y, color="blue", alpha=0.6, label="Données réelles")
sns.lineplot(x=X_test_s["RM"], y=y_pred_s, color="red", label="Régression")
plt.title("Régression linéaire simple : RM vs MEDV")
plt.xlabel("Nombre moyen de pièces (RM)")
plt.ylabel("Prix médian des maisons (MEDV)")
plt.legend()
plt.show()

# (b) Relation entre LSTAT et MEDV
plt.figure(figsize=(7,5))
sns.scatterplot(x=X["LSTAT"], y=y, color="green", alpha=0.6)
plt.title("Relation entre LSTAT et MEDV")
plt.xlabel("% de population à statut inférieur (LSTAT)")
plt.ylabel("Prix médian des maisons (MEDV)")
plt.show()

# (c) Valeurs réelles vs Prédictions (modèle multiple)
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test_m, y=y_pred_m, color="purple", alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Idéal")
plt.xlabel("Valeurs réelles (MEDV)")
plt.ylabel("Prédictions (MEDV)")
plt.title("Prédictions vs Réalité (Régression multiple)")
plt.legend()
plt.show()
