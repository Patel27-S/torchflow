import torch

x = torch.tensor(3.0, requires_grad=True)

# Define a function: y = x^2 + 2x + 1
y = x**2 + 2 * x + 1

# Backpropagate to compute gradients
y.backward()

# Print the gradient (dy/dx)
print("Gradient:", x.grad)  # dy/dx = 2x + 2 = 8 (when x = 3)


x = torch.tensor(2.0, requires_grad=True)

y = x**3
y.backward(retain_graph=True)  # Retain the computation graph
print("Gradient after first backward:", x.grad)

y.backward()  # Call backward again
print("Gradient after second backward:", x.grad)  # Accumulated gradient

x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()

print("Gradient before zeroing:", x.grad)
x.grad.zero_()  # Reset gradient to zero
print("Gradient after zeroing:", x.grad)
