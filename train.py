import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet


# Load intents data
with open("intents.json", "r") as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []

# Process training data
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        all_words.extend(words)
        xy.append((words, tag))


# Remove punctuation and duplicates
ignore_words = ["?", "!", ".", ","]

all_words = [
    stem(w.lower())
    for w in all_words
    if w not in ignore_words
]

all_words = sorted(set(all_words))
tags = sorted(set(tags))


# Create training data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)


# Create Dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(
            X_train, dtype=torch.float32
        )
        self.y_data = torch.tensor(
            y_train, dtype=torch.long
        )

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 8
hidden_size = 8
learning_rate = 0.001
num_epochs = 1000

train_dataset = ChatDataset()

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)


# Initialize neural network
input_size = len(X_train[0])
output_size = len(tags)

model = NeuralNet(
    input_size,
    hidden_size,
    output_size
)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# Training loop
for epoch in range(num_epochs):

    for words, labels in train_loader:

        outputs = model(words)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Loss: {loss.item():.4f}"
        )


# Save trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

torch.save(data, "data.pth")

print("\nTraining completed successfully!")
print("Model saved as data.pth")