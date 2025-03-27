# 2. Write a program to generate n-grams in reverse order (e.g., starting from the end of the sentence).

import nltk
from nltk.util import ngrams

nltk.download("punkt")


def generate_reverse_ngrams(text, n):
    words = nltk.word_tokenize(text.lower())
    words.reverse()

    reversed_n_grams = list(ngrams(words, n))
    return reversed_n_grams


text = "This is a sample text for n-gram generation."
result = generate_reverse_ngrams(text, 2)
print(result)
