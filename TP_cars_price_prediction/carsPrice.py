#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prédiction des Prix d'Automobiles - Script Exécutable
========================================================

Ce script implémente une solution complète pour prédire les prix d'automobiles
en utilisant des techniques de machine learning.

Auteur: Assistant Claude
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import warnings

# Configuration
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')

def load_and_explore_data(file_path):
    """Charger et explorer le dataset"""
    print("=== CHARGEMENT ET EXPLORATION DES DONNÉES ===")
    
    # Chargement
    df = pd.read_csv(file_path, index_col=0)
    print(f"✓ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
    # Informations générales
    print(f"\nColonnes disponibles: {list(df.columns)}")
    print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
    
    # Statistiques de base sur les prix
    print(f"\n=== STATISTIQUES DES PRIX ===")
    print(f"Prix moyen: ${df['price'].mean():.2f}")
    print(f"Prix médian: ${df['price'].median():.2f}")
    print(f"Prix min: ${df['price'].min():.2f}")
    print(f"Prix max: ${df['price'].max():.2f}")
    
    return df

def preprocess_data(df):
    """Prétraitement des données"""
    print("\n=== PRÉTRAITEMENT DES DONNÉES ===")
    
    df_processed = df.copy()
    
    # Variables catégorielles à encoder
    categorical_features = ['company', 'body-style', 'engine-type', 'num-of-cylinders']
    label_encoders = {}
    
    # Encodage
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[f'{feature}_encoded'] = le.fit_transform(df_processed[feature])
        label_encoders[feature] = le
        print(f"✓ {feature} encodé: {len(le.classes_)} classes")
    
    # Sélection des features
    feature_columns = ['wheel-base', 'length', 'horsepower', 'average-mileage'] + \
                      [f'{feat}_encoded' for feat in categorical_features]
    
    X = df_processed[feature_columns]
    y = df_processed['price']
    
    print(f"\n✓ Features sélectionnées: {len(feature_columns)}")
    print(f"✓ Shape des données: X={X.shape}, y={y.shape}")
    
    return X, y, feature_columns, label_encoders

def train_and_evaluate_models(X, y):
    """Entraîner et évaluer plusieurs modèles"""
    print("\n=== ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES ===")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèles à tester
    models = {
        'Régression Linéaire': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Arbre de Décision': DecisionTreeRegressor(random_state=42),
        'Forêt Aléatoire': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n Entraînement: {name}")
        
        # Entraînement
        if name in ['Régression Linéaire', 'Ridge', 'Lasso']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Évaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"   RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")
    
    return results, X_test, y_test, scaler, X_train_scaled, X_test_scaled

def analyze_best_model(results):
    """Analyser le meilleur modèle"""
    print("\n=== ANALYSE DU MEILLEUR MODÈLE ===")
    
    # Identifier le meilleur modèle
    best_model_name = max(results.keys(), key=lambda x: results[x]['R²'])
    best_result = results[best_model_name]
    
    print(f" Meilleur modèle: {best_model_name}")
    print(f" R² Score: {best_result['R²']:.4f}")
    print(f" RMSE: {best_result['RMSE']:.2f}")
    print(f" MAE: {best_result['MAE']:.2f}")
    
    return best_model_name, best_result

def create_prediction_function(best_model, best_model_name, feature_columns, 
                             label_encoders, scaler):
    """Créer une fonction de prédiction"""
    
    categorical_features = ['company', 'body-style', 'engine-type', 'num-of-cylinders']
    
    def predict_car_price(company, body_style, wheel_base, length, engine_type, 
                         num_cylinders, horsepower, average_mileage):
        """
        Prédire le prix d'une voiture basé sur ses caractéristiques
        """
        # Préparation des données
        new_data = {
            'company': company,
            'body-style': body_style,
            'wheel-base': wheel_base,
            'length': length,
            'engine-type': engine_type,
            'num-of-cylinders': num_cylinders,
            'horsepower': horsepower,
            'average-mileage': average_mileage
        }
        
        # Encodage des variables catégorielles
        for feature in categorical_features:
            try:
                encoded_val = label_encoders[feature].transform([new_data[feature]])[0]
                new_data[f'{feature}_encoded'] = encoded_val
            except ValueError:
                print(f"Attention: '{new_data[feature]}' non reconnu pour {feature}, utilisation de 0")
                new_data[f'{feature}_encoded'] = 0
        
        # Préparation des features
        X_new = np.array([[new_data[col] for col in feature_columns]])
        
        # Prédiction
        if best_model_name in ['Régression Linéaire', 'Ridge', 'Lasso']:
            X_new_scaled = scaler.transform(X_new)
            prediction = best_model.predict(X_new_scaled)[0]
        else:
            prediction = best_model.predict(X_new)[0]
        
        return max(0, prediction)  # Prix ne peut pas être négatif
    
    return predict_car_price

def display_results_summary(results):
    """Afficher un résumé des résultats"""
    print("\n" + "="*60)
    print("                 RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    
    # Créer un DataFrame des résultats
    results_df = pd.DataFrame({
        name: {
            'RMSE': scores['RMSE'],
            'MAE': scores['MAE'], 
            'R²': scores['R²']
        }
        for name, scores in results.items()
    }).T.round(4)
    
    results_df = results_df.sort_values('R²', ascending=False)
    print(results_df)
    
    print("\n Interprétation:")
    best_r2 = results_df['R²'].iloc[0]
    print(f"   • Le meilleur modèle explique {best_r2*100:.1f}% de la variance des prix")
    print(f"   • Erreur moyenne absolue: ${results_df['MAE'].iloc[0]:.0f}")
    print(f"   • Erreur quadratique moyenne: ${results_df['RMSE'].iloc[0]:.0f}")

def main():
    """Fonction principale"""
    print(" PRÉDICTION DES PRIX D'AUTOMOBILES")
    print("="*50)
    
    try:
        # 1. Chargement et exploration
        df = load_and_explore_data('./Automobiles.csv')
        
        # 2. Prétraitement
        X, y, feature_columns, label_encoders = preprocess_data(df)
        
        # 3. Entraînement et évaluation
        results, X_test, y_test, scaler, X_train_scaled, X_test_scaled = train_and_evaluate_models(X, y)
        
        # 4. Analyse du meilleur modèle
        best_model_name, best_result = analyze_best_model(results)
        best_model = best_result['model']
        
        # 5. Affichage du résumé
        display_results_summary(results)
        
        # 6. Création de la fonction de prédiction
        predict_price = create_prediction_function(
            best_model, best_model_name, feature_columns, label_encoders, scaler
        )
        
        # 7. Exemples de prédictions
        print("\n=== EXEMPLES DE PRÉDICTIONS ===")
        
        # Exemple 1: BMW sedan
        price1 = predict_price(
            company='bmw',
            body_style='sedan', 
            wheel_base=105.0,
            length=180.0,
            engine_type='ohc',
            num_cylinders='six',
            horsepower=150,
            average_mileage=20
        )
        print(f"\n  BMW Sedan Premium:")
        print(f"    Prix prédit: ${price1:,.2f}")
        
        # Exemple 2: Honda économique
        price2 = predict_price(
            company='honda',
            body_style='hatchback',
            wheel_base=95.0,
            length=160.0,
            engine_type='ohc',
            num_cylinders='four',
            horsepower=90,
            average_mileage=35
        )
        print(f"\n Honda Hatchback Économique:")
        print(f"    Prix prédit: ${price2:,.2f}")
        
        print("\n" + "="*50)
        print(" ANALYSE TERMINÉE AVEC SUCCÈS !")
        print("="*50)
        
        return predict_price, results, best_model_name
        
    except Exception as e:
        print(f"\n Erreur lors de l'exécution: {e}")
        return None, None, None

if __name__ == "__main__":
    # Exécution du script principal
    predict_function, all_results, best_model = main()
    
    # Interface simple pour tester des prédictions
    if predict_function:
        print("\n TESTEZ VOS PROPRES PRÉDICTIONS !")
        print("Utilisez la fonction predict_function() avec les paramètres suivants:")
        print("predict_function(company, body_style, wheel_base, length,")
        print("                engine_type, num_cylinders, horsepower, average_mileage)")
        print("\nExemples de valeurs valides:")
        print("- company: 'bmw', 'audi', 'honda', 'toyota', 'chevrolet', etc.")
        print("- body_style: 'sedan', 'hatchback', 'wagon', 'convertible'")
        print("- engine_type: 'ohc', 'dohc', 'ohcv', 'l'")
        print("- num_cylinders: 'three', 'four', 'five', 'six'")