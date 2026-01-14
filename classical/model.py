import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# training consts
NUM_EPOCHS = 50
BATCH_SIZE = 128

# Define model architecture
class GenreClassifier(nn.Module):
    def __init__(self, input_size=10, hidden1=256, hidden2=128, hidden3=64, num_classes=114):
        super(GenreClassifier, self).__init__()

        # Defining layers!

        # 10 features in -> 128 neurons hidden layer
        self.fc1 = nn.Linear(input_size, hidden1) # fully connected layer 1
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden2, hidden3)
        self.drop3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(hidden3, num_classes)
        
        self.relu = nn.ReLU()
    
    # forward pass
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        # note: no softmax here because Cross Entropy Loss expects raw logits
        return x
    
model = GenreClassifier(input_size=10, hidden1=256, hidden2=128, hidden3=64, num_classes=114)
criterion = nn.CrossEntropyLoss() # combines softmax + negative log likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === TRAINING ===

# load data into memory
X_train_scaled = np.load('data/processed/X_train_scaled.npy')
X_test_scaled = np.load('data/processed/X_test_scaled.npy')
X_val_scaled = np.load('data/processed/X_val_scaled.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')
y_val = np.load('data/processed/y_val.npy')

# load artifacts into memory
scaler = joblib.load('artifacts/scaler.pkl')
label_encoder = joblib.load('artifacts/label_encoder.pkl')

# verify data loaded correctly
print(f"X_train shape: {X_train_scaled.shape}")
print(f"X_test shape:  {X_test_scaled.shape}")
print(f"X_val shape:   {X_val_scaled.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
print(f"y_val shape:   {y_val.shape}")
print(f"Number of genres: {len(label_encoder.classes_)}")
print("Data Loaded Successfully")

# custom dataset class
class SpotifyDataset(Dataset):
    def __init__(self, X, y):
        # expecting np arrays
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

# datasets
training_dataset = SpotifyDataset(X_train_scaled, y_train)
val_dataset = SpotifyDataset(X_val_scaled, y_val)
test_dataset = SpotifyDataset(X_test_scaled, y_test)

# dataloaders
train_loader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # shuffles is false for consistent eval
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# training
print("=== TRAINING! ===")
print(f"Batch size: {BATCH_SIZE}")
print(f"Training batches: {len(train_loader)}")                                                                                                                                                        
print(f"Validation batches: {len(val_loader)}")  

# objects for logging
best_val_loss = float('inf')
history = {'training_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for X_batch, y_batch in train_bar:
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backwards pass
        optimizer.zero_grad() # QUESTION: why zero grad?
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += y_batch.size(0)
        train_correct += (predicted == y_batch).sum().item()
    # Calculate epoch metrics
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = train_correct / train_total

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # disables gradiant calculation
    with torch.no_grad():

        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Validation]')

        for X_batch, y_batch in val_bar:
            # only forwards
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

    # Log metrics to history
    history['training_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)

    # Print epoch summary
    print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}:')
    print(f'  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}')
    print(f'  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}')

    # Model checkpointing
    if avg_val_loss < best_val_loss:
        MOD_PATH = "artifacts/best_model_weights.pt"
        torch.save(model.state_dict(), MOD_PATH)
        best_val_loss = avg_val_loss
        print(f"saving better model at epoch {epoch+1}!")

# === POST-TRAINING SUMMARY ===
print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print(f"\nBest Validation Loss: {best_val_loss:.4f}")
print(f"Final Training Loss:  {history['training_loss'][-1]:.4f}")
print(f"Final Training Acc:   {history['train_acc'][-1]:.4f}")
print(f"Final Validation Acc: {history['val_acc'][-1]:.4f}")

# Save training history to JSON
history_path = "artifacts/training_history.json"
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"\nTraining history saved to: {history_path}")
print(f"Best model weights saved to: {MOD_PATH}")
