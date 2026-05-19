import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def set_custom_style(theme='light'):
    if theme == 'dark':
        plt.style.use('dark_background')
    else:
        plt.style.use('seaborn-v0_8-whitegrid')
    print(f"Style '{theme}' activé ")

def plot_generic_trends(df, x_col, y_col, title="Tendance"):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df[x_col], df[y_col], color='#1DB954')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, columns, title="Matrice de Corrélation"):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                vmin=-1, vmax=1, ax=ax)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_bivariate_scatter(df, x_col, y_col, hue_col=None, title="Nuage de Points"):
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df[x_col], df[y_col],
                        c=df[hue_col].astype('category').cat.codes if hue_col else '#1DB954',
                        alpha=0.5, cmap='viridis')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if hue_col:
        plt.colorbar(scatter, ax=ax, label=hue_col)
    plt.tight_layout()
    plt.show()

def plot_popularity_by_genre(df, title="Popularité par Genre"):
    fig, ax = plt.subplots(figsize=(14, 6))
    genre_pop = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)
    genre_pop.plot(kind='bar', color='#1DB954', ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Genre")
    ax.set_ylabel("Popularité moyenne")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_distribution(df, col, title=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[col], kde=True, color='#1DB954', ax=ax)
    ax.set_title(title or f"Distribution de {col}", fontsize=14)
    ax.set_xlabel(col)
    plt.tight_layout()
    plt.show()
