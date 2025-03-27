# 2. How would you use regular expressions to identify all phone numbers in a text document, regardless of their format (e.g., 123-456-7890, (123) 456-7890, etc.)?

import re


pattern = r'\+?\d{0,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

text = """
Here are some phone numbers:
123-456-7890
(123) 456-7890
123.456.7890
+1-123-456-7890
+91 9876543210
"""

result = re.findall(pattern, text)
print(result)
