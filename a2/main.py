import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X, weights, bias):
    z = X @ weights + bias  
    prediction = sigmoid(z) 
    return prediction

def backward(X, y, prediction):
    n = len(y)  # Number of samples
    
    # Calculate the gradient components
    error = y - prediction  # (y - ŷ)
    sigmoid_derivative = prediction * (1 - prediction)  # ŷ(1 - ŷ)
    
    # Combine them: (y - ŷ) ⊙ ŷ ⊙ (1 - ŷ)
    delta = error * sigmoid_derivative
    
    # Gradient for weights: -(2/n) · X^T · delta
    dW = -(2/n) * (X.T @ delta)
    
    # Gradient for bias: -(2/n) · sum(delta)
    db = -(2/n) * np.sum(delta)
    
    return dW, db

np.random.seed(42)

X = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=np.float32)

y = np.array([[0],
                  [0],
                  [0],
                  [1]], dtype=np.float32)


weights = np.random.randn(2, 1).astype(np.float32)
bias = np.random.randn(1).astype(np.float32) 

learning_rate = 0.5
num_epochs = 10000

# Training loop
for epoch in range(num_epochs):
    # 1. Forward pass
    prediction = forward(X, weights, bias)
    
    # 2. Calculate loss (MSE)
    loss = np.mean((y - prediction) ** 2)
    
    # 3. Backward pass (calculate gradients)
    dW, db = backward(X, y, prediction)
    
    # 4. Update weights and bias (gradient descent)
    weights -= learning_rate * dW
    bias -= learning_rate * db
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Final results
print("\nFinal predictions:")
final_pred = forward(X, weights, bias)
print(final_pred)
print("\nExpected:")
print(y)