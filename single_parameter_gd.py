import torch

# Initialize weight and learning rate
w = torch.tensor(2.0, requires_grad=True)  # Initial weight
learning_rate = 0.1

# Define a simple loss function: L = (w - 5)^2
for epoch in range(10):
    # Forward pass
    loss = (w - 5)**2  # Loss depends on weight

    # Backward pass
    loss.backward()  # Compute gradient

    # Update weight using gradient descent
    with torch.no_grad():  # Prevent gradient tracking for updates
        w -= learning_rate * w.grad

    # Zero gradients for next iteration
    w.grad.zero_()

    # Print progress
    print(f"Epoch {epoch+1}: Loss = {loss.item()}, Weight = {w.item()}")
