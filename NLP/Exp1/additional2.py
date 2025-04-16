#

import re

def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace (including tabs and newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Example text
text = "   This   is   a Sample   TEXT. \nNew line and \tTabs included!  "

normalized_text = normalize_text(text)

print("Original Text:\n", text)
print("\nNormalized Text:\n", normalized_text)
