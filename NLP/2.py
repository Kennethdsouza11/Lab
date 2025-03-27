import re


def clean_text(text):
    return re.sub(r"^\W+|\W+$", "", text)


text = "!Hello, word!"
result = clean_text(text)

print(result)
