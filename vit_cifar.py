import random
import numpy as np
import torch
import os

seed = 42
print(f"=====> Set the random seed as {seed}")
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(False)
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)

# simple reproducibility test
model = torch.nn.Linear(5, 5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(10):
    x = torch.randn(5, 5, requires_grad=True).cuda()
    y = model(x)
    y.sum().backward()
    optimizer.step()

print(y)