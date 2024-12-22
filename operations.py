import torch

# Arithmetic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b, end = '\n')  # Addition
print(a * b, end = '\n')  # Element-wise multiplication
print(torch.matmul(a, b))  # Dot product

# Reshaping
x = torch.tensor([1, 2, 3, 4])
y = x.reshape(2, 2)  # Converts to 2x2 matrix
z = x.view(2, 2)     # Another way to reshape