from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

# Initialize tokenizer and stemmer
tokenizer = TreebankWordTokenizer()
stemmer = PorterStemmer()


# Tokenization function
def tokenize(sentence):
    return tokenizer.tokenize(sentence)


# Stemming function
def stem(word):
    return stemmer.stem(word.lower())


# Bag of Words function
def bag_of_words(tokenized_sentence, all_words):
    import numpy as np

    sentence_words = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, word in enumerate(all_words):
        if word in sentence_words:
            bag[idx] = 1.0

    return bag