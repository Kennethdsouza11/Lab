# 4. Write a python program to extract phone numbers from the text in various formats and standardize them to a common format.

import re

def extract_and_standardize_phones(text):
    # Pattern to match phone numbers in formats like:
    # (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890, +91-9876543210
    pattern = r'(\+?\d{1,3})?[-.\s]?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})'

    matches = re.findall(pattern, text)

    standardized = []
    for match in matches:
        country_code = match[0] if match[0] else "+1"  # Default to +1 if no country code
        standardized.append(f"{country_code}-{match[1]}-{match[2]}-{match[3]}")

    return standardized

# Example text with mixed phone number formats
text = """
Contact us at (123) 456-7890 or 123-456-7890. 
Our international line is +1 123 456 7890 or +91-987-654-3210.
Also reach us at 123.456.7890 or 1234567890.
"""

result = extract_and_standardize_phones(text)
print("Standardized Phone Numbers:")
for phone in result:
    print(phone)
