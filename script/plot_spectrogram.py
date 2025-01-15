import matplotlib.pyplot as plt
import torch

spe = torch.jit.load("db.pt")
x = list(spe.parameters())[0]
plt.matshow(x.numpy())
plt.savefig("x.pdf")