from datasets import load_dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

SEED = 67

# fetch data
dataset = load_dataset("maharshipandya/spotify-tracks-dataset")                                 
df = pd.DataFrame(dataset['train'])  

print("num tracks: " + str(df.shape[0]))
print("num features: " + str(df.shape[1]))
print("num missing: " + str(df.isna().sum().sum()))

# Since there are only 3 max missing
df = df.dropna()
print("num missing after dropping NA: " + str(df.isna().sum().sum()))


# features we want to work with
feature_columns = ['danceability', 'energy', 'valence', 'loudness',                    
                  'speechiness', 'acousticness', 'instrumentalness',                  
                  'liveness', 'tempo', 'duration_ms']
X = df[feature_columns].values
y = df['track_genre'].values   

# convert genre labels to numbers! 0 - 113
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# First, 70 train 30 test
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=SEED)
# 15 test 15 validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# fit scaler on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X=X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Display mean and std for each feature
print("\n=== Scaler Statistics ===")
for i, feature_name in enumerate(feature_columns):
    print(f"{feature_name:20} | Mean: {scaler.mean_[i]:10.4f} | Std: {scaler.scale_[i]:10.4f}")                                     

# Save raw cleaned data
df.to_parquet('data/raw/spotify_cleaned.parquet', index=False)

# Save preprocessing artifacts (scaler, encoder)
joblib.dump(scaler, 'artifacts/scaler.pkl')
joblib.dump(label_encoder, 'artifacts/label_encoder.pkl')

# Save processed arrays to disk as .npy
np.save('data/processed/X_train_scaled.npy', X_train_scaled)
np.save('data/processed/X_test_scaled.npy', X_test_scaled)
np.save('data/processed/X_val_scaled.npy', X_val_scaled)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)
np.save('data/processed/y_val.npy', y_val)

print(f"\nSaved {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test, {X_val_scaled.shape[0]} val samples")