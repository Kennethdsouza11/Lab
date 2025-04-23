import torch
import torch.nn as nn

class HopfieldNetwork:
    def __init__(self):
        self.weight = None

    def train(self, patterns):
        # Convert to torch tensors
        patterns = torch.tensor(patterns, dtype=torch.float32)
        num_neurons = patterns.shape[1]
        self.weight = torch.zeros((num_neurons, num_neurons))

        for p in patterns:
            self.weight += torch.ger(p, p)  # Outer product

        # Zero the diagonal (no self-connection)
        self.weight.fill_diagonal_(0)

    def recall(self, pattern, steps=5):
        x = torch.tensor(pattern, dtype=torch.float32)
        for _ in range(steps):
            x = torch.sign(self.weight @ x)
        return x.int().tolist()

# Define binary patterns (use -1 and 1)
patterns = [
    [1, -1, 1, -1, 1],
    [-1, 1, -1, 1, -1]
]

# Initialize and train
hopfield = HopfieldNetwork()
hopfield.train(patterns)

# Test with a noisy version of first pattern
test = [1, -1, 1, 1, -1]
output = hopfield.recall(test)

print("Test Input :", test)
print("Recalled   :", output)
