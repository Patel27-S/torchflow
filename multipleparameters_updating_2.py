import torch

# Initialize a 3x3 matrix of parameters with requires_grad=True
W = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0], 
                  [7.0, 8.0, 9.0]], requires_grad=True)

# Target matrix T
T = torch.tensor([[1.5, 2.5, 3.5], 
                  [4.5, 5.5, 6.5], 
                  [7.5, 8.5, 9.5]])

# Learning rate
learning_rate = 0.1

# Training loop
for epoch in range(10):
    # Forward pass: Compute the loss
    loss = torch.sum((W - T)**2)  # Element-wise difference squared, then summed

    # Backward pass: Compute gradients
    loss.backward()

    # Update parameters using gradient descent
    with torch.no_grad():
        W -= learning_rate * W.grad

    # Zero gradients for the next iteration
    W.grad.zero_()

    # Print progress
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")
    print(f"W =\n{W}\n")
