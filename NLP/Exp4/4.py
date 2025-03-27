# WRITE A PROGRAM TO FIND OUT THE FREQUENCIES OF DISTINCT WORDS, GIVEN A SENTENCE USING N-GRAMS.

from collections import Counter
from nltk.util import ngrams


def word_frequencies(sentence, n):
    words = sentence.split()
    n_grams = ngrams(words, n)
    freq = Counter(n_grams)
    return freq


sentence = "this is a sample sentence this is "
n = 2
result = word_frequencies(sentence, n)
print(result)
