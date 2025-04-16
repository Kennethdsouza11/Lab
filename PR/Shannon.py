from collections import Counter
import math

def shannon_entropy(data):
    counter = Counter(data)
    total = len(data)
    
    entropy = 0.0
    for count in counter.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy

data = input('Enter the data (like ABABACCCCCAAB) : ')
entropy = shannon_entropy(data)
print(f"Shannon Entropy: {entropy:.4f}")