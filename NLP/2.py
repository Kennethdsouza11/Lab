# WRITE A PROGRAM TO REMOVE THE FIRST AND LAST CHARACTERS IF THEY ARE NOT LETTERS OR NUMBERS FROM A GIVEN SENTENCE.

import re


def clean_text(text):
    return re.sub(r"^\W+|\W+$", "", text)


text = "!Hello, word!"
result = clean_text(text)

print(result)
