import torch
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from torchaudio import load
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


class WavDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        audio, _ = load(self.filename)
        return audio


# Load the wav file
dataset = WavDataset("./sounds/7.wav")

# Create a StratifiedShuffleSplit object
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

# Split the dataset into training and validation sets
for train_index, val_index in splitter.split(dataset, dataset):
    X_train, X_val, y_train, y_val = dataset[train_index], dataset[val_index], dataset[train_index], dataset[val_index]

# Create DataLoaders for the training and validation sets
batch_size = 32
train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(X_val, batch_size=batch_size)

# Initialize and fine-tune the model
model = AudioMiniEncoderWithClassifierHead.from_pretrained("./tortoise/data/mel_norms.pth")

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_data in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = loss_fn(outputs, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_model.pth')
print("Fine-tuned model saved.")

# Evaluate the model on the validation set
model.eval()
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch_data in val_loader:
        outputs = model(batch_data)
        predicted = outputs.argmax(dim=1)
        correct_predictions += (predicted == y_val).sum().item()
        total_samples += y_val.size(0)

accuracy = correct_predictions / total_samples
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
