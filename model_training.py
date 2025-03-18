from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def prepare_data(df, target='EngagementLevel'):
    """Modelleme için özellikleri ve hedefi hazırlar"""
    X = df.drop(columns=[target, 'PlayerID'])
    y = df[target]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Veriyi eğitim ve test setlerine böler"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_models(X_train, y_train):
    """Farklı modelleri eğitir ve bir sözlük olarak döndürür"""
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\n{name} modeli eğitiliyor...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Tüm modellerin performansını değerlendirir ve görselleştirir"""
    model_results = []
    for name, model in models.items():
        print(f"\n{name} Model Değerlendirmesi:")
        y_pred = model.predict(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        print("Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred))
        
        # Karmaşıklık Matrisi
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", 
                    xticklabels=['Low', 'Medium', 'High'], 
                    yticklabels=['Low', 'Medium', 'High'])
        plt.title(f"{name} için Karmaşıklık Matrisi")
        plt.xlabel("Tahmin Edilen Etiketler")
        plt.ylabel("Gerçek Etiketler")
        plt.savefig(f'{name}_confusion_matrix.png')
        plt.close()
        
        # Sonuçları kaydet
        model_results.append({
            "Model": name,
            "Accuracy": accuracy
        })
    
    # Sonuç tablosu
    results_df = pd.DataFrame(model_results).sort_values(by="Accuracy", ascending=False)
    results_df.reset_index(drop=True, inplace=True)
    print("\nModel Değerlendirme Özeti:")
    print(results_df)
    
    # Bar plot ile model karşılaştırması
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=results_df, palette='Set3')
    plt.title('Modellerin Doğruluk Karşılaştırması')
    plt.ylim(0, 1)
    for i, v in enumerate(results_df['Accuracy']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.savefig('model_comparison.png')
    plt.close()
    
    return results_df

def save_models(models):
    """Eğitilmiş modelleri kaydeder"""
    for name, model in models.items():
        filename = f'{name}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} modeli {filename} olarak kaydedildi")