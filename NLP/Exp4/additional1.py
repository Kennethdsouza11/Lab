# 1. Write a program to calculate the probabilities of each n-gram in a sentence or text.

import nltk
from nltk.util import ngrams
from collections import Counter

nltk.download("punkt")


def calculate_ngram_probabilities(text, n):
    words = nltk.word_tokenize(text.lower())

    n_grams = list(ngrams(words, n))

    ngrams_count = Counter(n_grams)

    total_ngrams = sum(ngrams_count.values())

    ngrams_probability = {
        ngram: count / total_ngrams for ngram, count in ngrams_count.items()
    }
    return ngrams_probability


text = "This is a simple text. This text is just a sample"
n = 2
probability = calculate_ngram_probabilities(text, n)
print(probability)
