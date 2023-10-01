# import torch
# from tortoise.models import AudioMiniEncoderWithClassifierHead
# from torchaudio import load
# from sklearn.cluster import KMeans

# deepfake_voice_files = []
# for i in range(50):
#     deepfake_voice_files.append(load("deepfake_voice_files_{}.mp3".format(i), sr=19000))

# #cluster the deepfake files
# kmeans = KMeans(n_clusters=2)
# kmeans.fit(deepfake_voice_files)

# #labelling the deepfake files
# deepfake_labels = kmeans.labels_

# #split dataset into training and validation
# X_train, X_Val, y_train, y_val = torch.utils.data.random_split(
#     torch.cat(deepfake_voice_files), [int(0.75 * len(deepfake_voice_files)), int(0.25 * len(deepfake_voice_files))],
# )

# #fine tuning the model
# model = AudioMiniEncoderWithClassifierHead.from_pretrained("./tortoise/data/mel_norms.pth")

# #define the loss function and optimizer
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# #train the model for few epochs
# for epoch in range(10):
#     outputs = model(X_train)

#     loss = loss_fn(outputs, y_train)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# #evaluate the model's performance on validation set
# outputs = model(X_Val)

# #calculate accuracy
# accuracy = (outputs.argmax(dim=1) == y_val).float().mean()

# print("Accuracy: ", accuracy)

import torch
# from tortoise.models import AudioMiniEncoderWithClassifierHead
from tortoise import AudioMiniEncoderWithClassifierHead
from torchaudio import load
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

# Load and preprocess audio data
deepfake_voice_data = []
deepfake_labels = []

for i in range(50):
    audio, _ = load("deepfake_voice_files_{}.mp3".format(i), sr=19000)
    deepfake_voice_data.append(audio)
    # Assign labels based on the class (e.g., 0 for genuine, 1 for deepfake)
    deepfake_labels.append(1)  # You should label your data correctly

# Convert data and labels to PyTorch tensors
X = torch.stack(deepfake_voice_data)
y = torch.tensor(deepfake_labels)

# Split dataset into training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

# Create DataLoader for training and validation
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

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
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = loss_fn(outputs, batch_labels)
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
    for batch_data, batch_labels in val_loader:
        outputs = model(batch_data)
        predicted = outputs.argmax(dim=1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_samples += batch_labels.size(0)

accuracy = correct_predictions / total_samples
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Load the saved model for future use
loaded_model = AudioMiniEncoderWithClassifierHead.from_pretrained("./tortoise/data/mel_norms.pth")
loaded_model.load_state_dict(torch.load('fine_tuned_model.pth'))
loaded_model.eval()  # Put the loaded model in evaluation mode
