import torch

# 1. Start with a parameter we want to learn
x = torch.tensor(1.5, requires_grad=True)

# 2. Try to use it to create a hard mask (step function)
# If x > 1.0, mask is 1, else 0.
mask = (x > 1.0).float() 

# 3. Use the mask in some calculation (our "loss")
loss = mask * 5.0

# 4. Try to backpropagate
try:
    loss.backward()
    print("Gradient of x:", x.grad)
except Exception as e:
    print("Error during backward pass:", e)

# ---------------------------------------------
# Now let's try the differentiable way (Sigmoid)
# ---------------------------------------------
x2 = torch.tensor(1.5, requires_grad=True)

# Use sigmoid to create a soft mask
# We scale by a large number to make it steep like a step function
mask_soft = torch.sigmoid(10.0 * (x2 - 1.0))

loss_soft = mask_soft * 5.0

loss_soft.backward()
print("Gradient of x2 (Soft Mask):", x2.grad)

