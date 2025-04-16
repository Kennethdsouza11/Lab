import nltk

nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The striped bats were hanging on their feet for best"

tokens = nltk.word_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Original Word | Stemmed   | Lemmatized")
print("----------------------------------------")

for word in tokens:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    print(f"{word:13} | {stem:9} | {lemmatizer}")
