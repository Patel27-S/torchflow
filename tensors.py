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

# Element-wise addition
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("Addition:", a + b)

# Element-wise multiplication
print("Multiplication:", a * b)

# Matrix multiplication
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])
print("Matrix Multiplication:\n", torch.mm(matrix_a, matrix_b))
