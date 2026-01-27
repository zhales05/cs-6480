import torch
import torch.nn.functional as F

def compute_loss(prediction, y):
    return F.mse_loss(prediction, y)

def forward(X, weights, bias):
    return torch.sigmoid(X @ weights + bias)

X = torch.tensor([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=torch.float32)

y = torch.tensor([[0],
                  [0],
                  [0],
                  [1]], dtype=torch.float32)


torch.manual_seed(42) 

weights = torch.randn(2, 1, requires_grad=True)  # 2 inputs, 1 output
bias = torch.randn(1, requires_grad=True)  

learning_rate = 0.5
num_epochs = 10000

for epoch in range(num_epochs):
    prediction = forward(X, weights, bias)
    loss = compute_loss(prediction, y)
    # compute gradients
    loss.backward()
    
    with torch.no_grad():
        weights -= learning_rate * weights.grad
        bias -= learning_rate * bias.grad
        
    weights.grad.zero_()
    bias.grad.zero_()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    
print("\nFinal predictions:")
with torch.no_grad():
    final_pred = forward(X, weights, bias)
    print(final_pred)
print("\nExpected:")
print(y)


