# 1. Write a regex pattern to extract all dates in the format DD/MM/YYYY from a paragraph of text.

import re

pattern = r"\b(0[1-9]|[12][0-9]|3[0-1])/(0[1-9]|1[0-2])/[0-9]{4}\b"

text = "Today's date is 27/03/2025. The project started on 05/12/2023 and ended on 15/07/2024."

dates = re.findall(pattern, text)

print(["/".join(match) for match in dates])
