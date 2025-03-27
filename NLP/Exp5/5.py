# WRITE A PROGRAM TO REMOVE DIGITS FROM A GIVEN SENTENCE USING GREEDY TOKENIZER.

import nltk
from nltk.util import ngrams
import re

nltk.download("punkt")


def remove_digits(sentence):
    words = nltk.word_tokenize(sentence.lower())
    filtered_words = [re.sub(r"\d+", "", word) for word in words]
    return " ".join(filtered_words)


sentence = "The price is 100 dollars and the event is on 23rd June 2024"
result = remove_digits(sentence)
print(result)
