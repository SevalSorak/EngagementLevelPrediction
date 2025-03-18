import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def basic_statistics(df):
    """Temel istatistiksel özet oluşturur"""
    print("Veri Seti Bilgisi:")
    print(df.info())
    print("\nTemel İstatistikler:")
    print(df.describe())

def plot_categorical_distribution(df, column_name):
    """Kategorik değişkenlerin dağılımını görselleştirir"""
    plt.figure(figsize=(10, 4))
    
    # Çubuk grafik
    plt.subplot(1, 2, 1)
    ax = sns.countplot(y=column_name, data=df, palette='Set3')
    plt.title(f'{column_name} Dağılımı')
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2), 
                    ha='center', va='center', xytext=(10, 0), textcoords='offset points')
    sns.despine(left=True, bottom=True)
    
    # Pasta grafik
    plt.subplot(1, 2, 2)
    df[column_name].value_counts().plot.pie(
        autopct='%1.1f%%', 
        colors=sns.color_palette('Set3'), 
        startangle=90, 
        explode=[0.05] * df[column_name].nunique()
    )
    plt.title(f'{column_name} Yüzde Dağılımı')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig(f'{column_name}_distribution.png')
    plt.close()

def visualize_distributions(df):
    """Sayısal değişkenler için dağılım grafikleri oluşturur"""
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    set3_colors = sns.color_palette("Set3", len(numerical_cols))
    
    plt.figure(figsize=(16, 12))
    for i, column in enumerate(numerical_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[column], kde=True, bins=10, color=set3_colors[i-1])
        plt.title(f'{column.replace("_", " ")} Dağılımı')
        plt.xlabel(column.replace('_', ' '))
        plt.ylabel('Frekans')
    
    plt.tight_layout()
    plt.savefig('numerical_distributions.png')
    plt.close()

def correlation_analysis(df):
    """Korelasyon matrisi ve ısı haritası oluşturur"""
    numerical_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numerical_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', center=0, fmt='.2f')
    plt.title('Korelasyon Matrisi')
    plt.savefig('korelasyon_isi_haritasi.png')
    plt.close()
    
    return correlation_matrix

def visualize_all(df):
    """Tüm görselleştirmeleri çalıştırır"""
    categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
    for col in categorical_cols:
        plot_categorical_distribution(df, col)
    visualize_distributions(df)
    correlation_analysis(df)