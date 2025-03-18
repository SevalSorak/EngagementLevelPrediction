import pandas as pd
from sklearn.preprocessing import StandardScaler

def create_features(df):
    """Mevcut verilerden yeni özellikler oluşturur"""
    # Toplam oyun süresini dakikaya çevir
    df['TotalPlayTimeMinutes'] = df['PlayTimeHours'] * 60
    
    # Oturum başına ortalama başarı
    df['AchievementsPerSession'] = df['AchievementsUnlocked'] / (df['SessionsPerWeek'] + 1)
    
    return df

def encode_categorical(df):
    """Kategorik değişkenleri kodlar"""
    categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

def scale_features(df):
    """Sayısal özellikleri ölçeklendirir"""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler