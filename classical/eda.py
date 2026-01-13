# EDA for Track Genre Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12,6)

def load_cleaned_data():
    df = pd.read_parquet("./spotify_cleaned.parquet")
    print(f"Loaded {df.shape[0]} tracks with {df.shape[1]} features")
    return df

# 1. feature distribution analysis

# genre
def eda(df):
    # count genres
    genre_counts = df['track_genre'].value_counts()
    
      # Check how many are close to 1000                                                              
    around_1000 = genre_counts[(genre_counts >= 990) & (genre_counts <= 1010)]                      
    print(f"\nGenres with 990-1010 samples: {len(around_1000)} out of {len(genre_counts)}") 

    # all genres have 1000 tracks (except ones like kpop that were cleaned)
    
    # correlation heatmap
    audio_features = ['danceability', 'energy', 'valence', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'tempo', 'duration_ms']
    corr_matrix = df[audio_features].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Feature Correlations')
    plt.savefig('corr_matrix.png')

    # TODO: use corr matrix to determine features to add/remove
    # for now, just use all features

    # check feature scales
    for feature in audio_features:
        print(f"{feature:20s}: [{df[feature].min():10.2f}, {df[feature].max():10.2f}]")

    # some tracks have 0 tempo? investigate:
    zero_tempo = df[df['tempo'] == 0]
    print(f"{len(zero_tempo)} tracks WITH 0 bpm")
    print(zero_tempo['track_genre'].value_counts().head())  
    zero_tempo_notsleep = zero_tempo[zero_tempo['track_genre'] != 'sleep']
    print(f"{len(zero_tempo_notsleep)} nonsleep tracks WITH 0 bpm")

    # majority of them are in sleep, but keep all so that model doesn't automatically think 0bpm == sleep             
def main():
    df = load_cleaned_data()

    eda(df)


if __name__ == "__main__":
    main()