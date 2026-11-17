import random
import json
import torch

from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words


# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)


# Load trained model
data = torch.load("data.pth", map_location="cpu", weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]

all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data["model_state"])
model.eval()


print("AI Chatbot is ready!")
print("Type 'quit' to exit.\n")


# Interactive chat loop
while True:

    sentence = input("You: ")

    if sentence.lower() == "quit":
        print("Bot: Goodbye! Have a nice day!")
        break

    # Tokenize input
    sentence_words = tokenize(sentence)

    # Remove punctuation
    content_words = [
        word for word in sentence_words
        if any(char.isalnum() for char in word)
    ]

    # Check vocabulary matches
    known_words = set(all_words)

    matched_words = [
        stem(word) for word in content_words
        if stem(word) in known_words
    ]

    coverage = (
        len(set(matched_words)) / len(set(stem(w) for w in content_words))
        if content_words else 0
    )

    # Convert to Bag of Words
    X = bag_of_words(sentence_words, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    # Predict intent
    with torch.no_grad():
        output = model(X)
        probabilities = torch.softmax(output, dim=1)

        probability, predicted = torch.max(
            probabilities, dim=1
        )

        tag = tags[predicted.item()]

    # Confidence and vocabulary filtering
    if probability.item() > 0.75 and coverage >= 0.5:

        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print("Bot:", random.choice(intent["responses"]))
                break

    else:
        print("Bot: Sorry, I don't understand.")