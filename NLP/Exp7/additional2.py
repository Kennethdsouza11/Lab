# 2.Write a python program to create a Named Entity Recognition (NER) system to identify entities (like people, organizations, dates) and perform text normalization (remove or standardize text such as dates, monetary values, and numbers).

import re
import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function for NER + Normalization
def process_text(text):
    doc = nlp(text)

    print("\n🟢 Named Entities Found:")
    for ent in doc.ents:
        print(f"{ent.text:25} → {ent.label_}")

    # Normalize text
    normalized_text = text
    normalized_text = re.sub(r'\$?\d+(,\d{3})*(\.\d+)?', '[NUM]', normalized_text)  # Replace monetary/numeric values
    normalized_text = re.sub(r'\b\d{1,2}[-/th|st|nd|rd\s]*[A-Za-z]+[-/\s]*\d{2,4}\b', '[DATE]', normalized_text)  # Rough date
    normalized_text = re.sub(r'\b\d{4}\b', '[YEAR]', normalized_text)  # Years like 2024

    print("\n🧹 Normalized Text:")
    print(normalized_text)

# Example input
sample_text = """
Barack Obama was born on August 4, 1961. He served as the 44th President of the United States.
Apple Inc. reported revenue of $123.9 billion in Q1 2022. The event happened on 12/03/2022.
"""

# Run the function
process_text(sample_text)
