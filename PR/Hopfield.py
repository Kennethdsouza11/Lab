import torch
from hopfield import Hopfield, HopfieldPooling

stored_pattern = torch.tensor([
    [1,-1,1,-1],
    [-1,1,-1,1]
])

hopfield = Hopfield(
    input_size = 4,
    hidden_size = 4,
    scaling = 1,
    update_steps_max = 10,
    num_heads = 1,
    dropout = 0.0,
    sequence_length = 2
)

query = torch.tensor([
    [1,-1,-1,-1]
])

retrieved = hopfield(query = query, stored_memory = stored_pattern)

print(query)
print(torch.round(retrieved))
