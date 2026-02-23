import torch
import torch.nn as nn

# Let's say we have a learned logit for a rank slice
logit = torch.tensor(1.5, requires_grad=True)

# 1. Calculate probability using logistic regression (sigmoid)
prob = torch.sigmoid(logit) # prob = 0.817

# 2. Do a hard gating on 50%
mask = (prob > 0.5).float() # mask = 1.0

loss = mask * 5.0

try:
    loss.backward()
    print("Gradient of logit:", logit.grad)
except RuntimeError as e:
    print("Error:", e)
