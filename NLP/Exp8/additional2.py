import spacy
from spacy import displacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Barack Obama was born in Hawaii and served as the President of the United States."

# Process the text
doc = nlp(text)

# 1. Part of Speech Tagging
print("Part of Speech Tagging:\n")
for token in doc:
    print(f"{token.text:15} → {token.pos_}")

# 2. Dependency Parsing
print("\nDependency Parsing:\n")
for token in doc:
    print(f"{token.text:15} → {token.dep_:10} (Head: {token.head.text})")

# 3. Noun Phrase Extraction
print("\nNoun Phrases:\n")
for chunk in doc.noun_chunks:
    print(f"- {chunk.text}")

# 4. Named Entity Recognition (NER)
print("\nNamed Entities:\n")
for ent in doc.ents:
    print(f"{ent.text:25} → {ent.label_}")

# 5. Visualization (Dependency Tree)
print("\nLaunching Dependency Visualization in your browser...")
displacy.serve(doc, style="dep")
