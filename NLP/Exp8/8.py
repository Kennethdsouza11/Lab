# DEMONSTRATE OBJECT STANDARDIZATION SUCH AS REPLACE SOCIAL MEDIA SLANGS FROM A TEXT.
import re

# Dictionary mapping social media slang to standard English
slang_dict = {
    "brb": "be right back",
    "lol": "laughing out loud",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "idk": "I don't know",
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "lmk": "let me know",
    "smh": "shaking my head",
    "np": "no problem",
    "imo": "in my opinion"
}

# Sample input text with social media slang
text = "OMG! idk what to do lol. brb, ttyl. btw u r awesome! lmk if u need anything."

# Lowercase the text to handle case-insensitive slang
text = text.lower()

# Tokenize the text
tokens = re.findall(r'\b\w+\b', text)

# Replace slangs with standard equivalents
standardized_tokens = [slang_dict.get(word, word) for word in tokens]

# Join tokens to form standardized text
standardized_text = ' '.join(standardized_tokens)

# Print results
print("Original Text:")
print(text)

print("\nStandardized Text:")
print(standardized_text)
