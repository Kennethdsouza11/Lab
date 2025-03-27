# 1. Write a program to count how many digits are present in a given sentence.
import re


def count_digits(sentence):
    digits = re.findall(r"\d+", sentence)
    return len(digits)


sentence = "The price is 100 dollars and the event is on 23rd June 2024."
result = count_digits(sentence)
print(result)
