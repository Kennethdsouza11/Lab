# 2. Write a program to replace all characters that are not letters or numbers in a sentence with a specified character.

import re


def replace_character(sentence, replacement_char):
    return re.sub(r"[\W+]", replacement_char, sentence)


sentence = "Hello, World! 123 @#$%^&*()"
replacement_char = "*"
result = replace_character(sentence, replacement_char)
print(result)
