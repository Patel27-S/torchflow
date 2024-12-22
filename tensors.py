import torch

# Creating tensors
x = torch.tensor([1, 2, 3])  # 1D tensor
y = torch.tensor([[1, 2], [3, 4]])  # 2D tensor

# Common tensor creation methods
zeros = torch.zeros(2, 3)  # 2x3 tensor of zeros
ones = torch.ones(2, 3)   # 2x3 tensor of ones
rand = torch.rand(2, 3)   # 2x3 tensor of random numbers

print(rand)
print(ones)
print(zeros)