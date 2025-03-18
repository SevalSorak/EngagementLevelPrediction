import pandas as pd
import numpy as np

def load_data(file_path):
    """CSV dosyasından veri setini yükler"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Veri setini temizler: eksik değerleri ve veri tiplerini düzenler"""
    # Eksik değerleri kontrol et
    print("Eksik değerler:\n", df.isnull().sum())
    
    # Kategorik değişkenleri kategori tipine çevir
    categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    # Sayısal sütunların doğru formatta olduğundan emin ol
    numerical_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                     'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def remove_outliers(df):
    """IQR yöntemiyle aykırı değerleri kaldırır"""
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df