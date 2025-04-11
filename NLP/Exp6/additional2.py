#Write a python program to extract all dates in various formats (DD/MM/YYYY, MM-DD-YYYY, Month Day, Year) from mixed text.

import re

def extract_dates(text):
    # Regex patterns for different date formats
    pattern_ddmmyyyy = r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[0-2])/\d{4}\b'
    pattern_mmddyyyy = r'\b(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])-\d{4}\b'
    pattern_month_day_year = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b'

    # Combine all patterns
    combined_pattern = f'{pattern_ddmmyyyy}|{pattern_mmddyyyy}|{pattern_month_day_year}'

    # Find all matches
    matches = re.findall(combined_pattern, text)

    # Flatten the list and remove empty strings
    dates = ["".join(match).strip() for match in matches if any(match)]

    return dates

# Example text with mixed date formats
text = """
The event is on 25/03/2024. Another one happened on 03-14-2023. 
We also remember March 18, 2022 and April 1, 2020 as important dates.
"""

extracted_dates = extract_dates(text)
print("Extracted Dates:")
for date in extracted_dates:
    print("-", date)

