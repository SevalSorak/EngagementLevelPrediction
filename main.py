import pandas as pd
from data_preprocessing import load_data, clean_data, remove_outliers
from exploratory_analysis import basic_statistics, visualize_all
from feature_engineering import create_features, encode_categorical, scale_features
from model_training import prepare_data, split_data, train_models, evaluate_models, save_models

def main():
    # Veriyi yükle ve ön işlemleri yap
    df = load_data('online_gaming_behavior_dataset.csv')  # Veriyi önce CSV olarak kaydet
    df_cleaned = clean_data(df)
    df_no_outliers = remove_outliers(df_cleaned)
    
    # Keşifsel Veri Analizi
    basic_statistics(df_no_outliers)
    visualize_all(df_no_outliers)  # Güncellenmiş görselleştirme fonksiyonu
    
    # Özellik Mühendisliği
    df_features = create_features(df_no_outliers)
    df_encoded = encode_categorical(df_features)
    df_scaled, scaler = scale_features(df_encoded)
    
    # Modelleme için veriyi hazırla
    X, y = prepare_data(df_scaled)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Modelleri eğit ve değerlendir
    models = train_models(X_train, y_train)
    evaluate_models(models, X_test, y_test)
    
    # Modelleri kaydet
    save_models(models)

if __name__ == "__main__":
    main()