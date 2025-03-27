# 1. Write a program to count the number of characters that are not letters or numbers in a given sentence.

import re


def count_non_alphanumeric(sentence):
    non_alnum_chars = re.findall(r"[\W+]", sentence)
    return len(non_alnum_chars)


sentence = "Hello, World! 123 @#$%^&*()"
print(count_non_alphanumeric(sentence))
