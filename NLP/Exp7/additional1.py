# 1.	Write a python program to design a custom tokenizer that can handle multiple types of noise (hashtags, mentions, URLs, punctuation, etc.) and then apply stemming and lemmatization.

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

# Initialize tools
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# Define custom tokenizer
def custom_tokenizer(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove hashtags and mentions
    text = re.sub(r"#\w+|@\w+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and empty tokens
    tokens = [word for word in tokens if word not in stop_words and word.strip()]

    return tokens


# Sample text to process
sample_text = (
    "Hey @user! Check out https://example.com #awesome 😎. We're loving AI in 2024!!"
)

# Tokenize and clean
tokens = custom_tokenizer(sample_text)

# Display results
print(f"{'Word':12} | {'Stemmed':12} | {'Lemmatized'}")
print("-" * 40)
for word in tokens:
    stemmed = stemmer.stem(word)
    lemmatized = lemmatizer.lemmatize(word)
    print(f"{word:12} | {stemmed:12} | {lemmatized}")
