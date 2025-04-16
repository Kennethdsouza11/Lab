import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Sample dataset (you can expand this!)
data = [
    ("The game was exciting and the team played well", "sports"),
    ("The election results were announced by the government", "politics"),
    ("He scored a goal in the last minute", "sports"),
    ("The parliament passed a new law", "politics")
]

# Download required NLTK resources
nltk.download("punkt")

# Step 1: Create a vocabulary and encode words
vocab = set()
for sentence, _ in data:
    tokens = word_tokenize(sentence.lower())
    vocab.update(tokens)
vocab = list(vocab)
word_to_index = {word: idx for idx, word in enumerate(vocab)}

# Step 2: Group data by class
class_data = defaultdict(list)
for sentence, label in data:
    tokens = word_tokenize(sentence.lower())
    encoded = [word_to_index[word] for word in tokens]
    class_data[label].append(np.array(encoded))

# Step 3: Train an HMM for each class
models = {}
for label, sequences in class_data.items():
    lengths = [len(seq) for seq in sequences]
    X = np.concatenate(sequences).reshape(-1, 1)
    model = hmm.MultinomialHMM(n_components=4, n_iter=100)
    model.fit(X, lengths)
    models[label] = model

# Step 4: Prediction function
def predict(text):
    tokens = word_tokenize(text.lower())
    encoded = [word_to_index.get(word, 0) for word in tokens]
    X = np.array(encoded).reshape(-1, 1)
    scores = {label: model.score(X) for label, model in models.items()}
    return max(scores, key=scores.get)

# Step 5: Test predictions
test_texts = [
    "The striker hit a great shot",
    "The president addressed the nation"
]

for test in test_texts:
    label = predict(test)
    print(f"Text: {test}\nPredicted Label: {label}\n")
