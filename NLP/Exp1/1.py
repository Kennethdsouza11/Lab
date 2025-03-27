# WRITE A PROGRAM GIVEN A PIECE OF TEXT, WE WANT TO SPLIT THE TEXT AT ALL SPACES (INCLUDING NEW LINE CHARACTERS AND CARRIAGE RETURNS) AND PUNCTUATION MARKS.

import re


def split_text(text):
    words = re.split(r"[\s\W]+", text)
    words = [word for word in words if word]
    return words


text = "Hello, world!\nThis is a test. Let's see how it works."
result = split_text(text)
print(result)
