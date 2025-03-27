# 2. Write a program to split a word into two parts at random positions and print all splits.
def generate_splits(word):
    return [[word[:i], word[i:]] for i in range(1, len(word))]


word = "hello"
print(generate_splits(word))
