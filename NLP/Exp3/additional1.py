# Write a program to generate all possible prefixes and suffixes of a given word.

def generate_prefixes_suffixes(word):
    prefixes = [word[:i] for i in range(1, len(word) + 1)]
    suffixes = [word[i:] for i in range(len(word))]
    return prefixes, suffixes

word = "hello"
result = generate_prefixes_suffixes(word)
print(result)
