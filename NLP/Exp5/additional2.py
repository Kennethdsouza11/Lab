# 2. Write a program to extract and print all digits from a sentence using a greedy tokenizer.
import nltk

nltk.download("punkt")
import re


def extract_digits(sentence):
    words = nltk.word_tokenize(sentence.lower())
    digits = [re.sub(r"\D", "", word) for word in words if re.search(r"\d", word)]
    digits = [num for num in digits if num]
    return digits


sentence = "The price is 100 dollars and the event is on 23rd June 2024."
extracted_digits = extract_digits(sentence)
print(extracted_digits)
