# Write a python program to perform part of speech tagging on any textual data.

import nltk
from nltk.tokenize import word_tokenize

# Download necessary resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Sample text (you can change this to any text you want)
text = "The quick brown fox jumps over the lazy dog."

# Tokenize the text into words
tokens = word_tokenize(text)

# Perform POS tagging
pos_tags = nltk.pos_tag(tokens)

# Print the POS tags
print("Part of Speech Tagging:\n")
for word, tag in pos_tags:
    print(f"{word:12} → {tag}")
