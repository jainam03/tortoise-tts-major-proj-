import tensorflow as tf
from tensorflow.keras.utils import load_audio
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedShuffleSplit


class WavDataset(tf.keras.utils.DatasetV2):
    def __init__(self, filename):
        self.filename = filename

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        audio = load_audio(self.filename)
        return audio


# Load the wav file
dataset = WavDataset("./sounds/7.wav")

# Create a StratifiedShuffleSplit object
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)

# Split the dataset into training and validation sets
for train_index, val_index in splitter.split(dataset, dataset):
    X_train, X_val, y_train, y_val = dataset[train_index], dataset[val_index], dataset[train_index], dataset[val_index]

# Create DataLoaders for the training and validation sets
batch_size = 10
train_loader = tf.keras.utils.Sequence(X_train, batch_size=batch_size)
val_loader = tf.keras.utils.Sequence(X_val, batch_size=batch_size)

# Define the model
model = tf.keras.models.Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_loader, epochs=5)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_loader)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.2f}%")
