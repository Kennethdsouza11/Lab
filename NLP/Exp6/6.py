# DEMONSTRATE NOISE REMOVAL FOR ANY TEXTUAL DATA AND REMOVE REGULAR EXPRESSION PATTERN SUCH AS HASHTAG FROM TEXTUAL DATA.
import re

def clean_text(text):
    # Remove hashtags (e.g., #AI)
    text = re.sub(r"#\w+", "", text)

    # Remove mentions (e.g., @elonmusk)
    text = re.sub(r"@\w+", "", text)

    # Remove URLs (e.g., https://example.com)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters except letters, digits, and spaces
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Example noisy text
noisy_text = "Check this out! #AI is the future. Visit https://ai.example.com 😎 @user123"

# Cleaned text
cleaned = clean_text(noisy_text)
print("Original Text:\n", noisy_text)
print("\nCleaned Text:\n", cleaned)
