import torch

# Initialize parameters and learning rate
w1 = torch.tensor(0.0, requires_grad=True)  # Initial value for w1
w2 = torch.tensor(0.0, requires_grad=True)  # Initial value for w2
learning_rate = 0.1

# Training loop
for epoch in range(10):
    # Forward pass: Compute the loss
    loss = (w1 - 3)**2 + (w2 + 2)**2

    # Backward pass: Compute gradients
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

    # Zero gradients for next iteration
    w1.grad.zero_()
    w2.grad.zero_()

    # Print progress
    print(f"Epoch {epoch+1}: Loss = {loss.item()}, w1 = {w1.item()}, w2 = {w2.item()}")
