import torch
N = 3
T = torch.randn(2, N)
A = torch.randn(N)

result = torch.cat((T, A[None]), dim=0)
print(result)