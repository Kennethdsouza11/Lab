# 3. Write a program that greedily tokenizes a sentence but prioritizes specific patterns, such as dates (\d{1,2}/\d{1,2}/\d{2,4}) and email addresses ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}), over general tokens.
import re
import nltk

nltk.download("punkt")


def prioritize_sentence(sentence):
    patterns = [
        (r"\d{1,2}/\d{1,2}/\d{2,4}"),
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    ]

    tokens = []
    processed_sentence = sentence

    for pattern, label in patterns:
        matches = re.findall(pattern, processed_sentence)
        for match in matches:
            tokens.append(match)
            processed_sentence = processed_sentence.replace(match, "")
        remaining_tokens = nltk.word_tokenize(processed_sentence)
        tokens.append(remaining_tokens)
        return tokens
