# Write a program to detect and remove emoticons or emojis from textual data.

import re
import emoji

def remove_emojis_emoticons(text):
    # Remove emojis using the emoji module
    text = emoji.replace_emoji(text, replace='')

    # Remove common emoticons using regex
    emoticon_pattern = r'(:\)|:\(|:D|:P|;D|;\)|:-\)|:-\(|:-D|:-P|:O|:\'\(|<3)'
    text = re.sub(emoticon_pattern, '', text)

    return text.strip()

# Example text containing emojis and emoticons
text = "I'm so happy today! 😄❤️ Let's go out! :)"

cleaned_text = remove_emojis_emoticons(text)

print("Original Text:\n", text)
print("\nCleaned Text:\n", cleaned_text)