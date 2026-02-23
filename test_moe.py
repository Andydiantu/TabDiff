import torch

# Let's say effective_r is 3.5, so probabilities are:
# Index 0, 1, 2, 3 have p > 0.5 (active) -> Hard Gate = 1.0
# Index 4, ... have p < 0.5 (inactive) -> Hard Gate = 0.0
prob_mask = torch.tensor([0.99, 0.95, 0.8, 0.2, 0.05], requires_grad=True)

# 1. Hard Zero Gating
hard_gate = (prob_mask > 0.5).float() # [1., 1., 1., 0., 0.]

# THE TRICK: We multiply the Hard Gate by the Probability!
gated_probs = prob_mask * hard_gate 
print("Gated Probs: ", gated_probs)

loss = (gated_probs * 5.0).sum()
loss.backward()

print("With Trick Grad:", prob_mask.grad)
