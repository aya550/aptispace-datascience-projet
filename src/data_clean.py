import pandas as pd
import numpy as np
from typing import List

def load_raw_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Données chargées avec succès. Dimensions : {df.shape}")
        return df
    except Exception as e:
        print(f" Erreur lors du chargement : {e}")
        raise e

def clean_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    print(f"Dates nettoyées dans '{date_col}'.")
    return df

def handle_outliers(df: pd.DataFrame, columns: List[str], min_val: float, max_val: float) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        outlier_mask = (df[col] < min_val) | (df[col] > max_val)
        df.loc[outlier_mask, col] = np.nan
        print(f" Outliers dans '{col}' remplacés par NaN")
    return df

def impute_missing_values(df: pd.DataFrame, columns: List[str], method: str = 'interpolate') -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if method == 'interpolate':
            df[col] = df[col].interpolate(method='linear')
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].ffill()
    print(f"Valeurs manquantes imputées avec '{method}'")
    return df

def feature_engineering(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df['hour'] = df[date_col].dt.hour
        df['dayofweek'] = df[date_col].dt.dayofweek
        print(" Colonnes 'hour' et 'dayofweek' ajoutées")
    else:
        print(f" '{date_col}' n'est pas au format Datetime.")
    return df