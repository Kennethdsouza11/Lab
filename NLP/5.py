# WRITE A PROGRAM TO REMOVE DIGITS FROM A GIVEN SENTENCE USING GREEDY TOKENIZER.

import re


def remove_digits(sentence):
    return re.sub(r"[\d+]", "", sentence)


sentence = "This is a sample 234d test. It contains 35434sd4r letters"
result = remove_digits(sentence)
print(result)
