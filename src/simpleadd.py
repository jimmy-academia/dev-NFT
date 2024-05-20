import torch
from utils import *
s = torch.Tensor([0.1, 0.1, 0.1, 0.1])
# s = torch.Tensor([0.2, 0.2, 0.2, 0.4])
B = 7
# p = torch.Tensor([15.5, 7.4, 8.3]) 
p = torch.Tensor([15.5, 7.4, 8.3]) + torch.Tensor([-0.869, -0.317, -0.447]) * 5
print(p)

V = torch.Tensor([0.653, 0.653, 0.954])

z = torch.ones(3)


for __ in range(1000):
    s_var = s.clone()
    s_var.requires_grad = True
    x = s_var[:-1]*B/p
    U1 = (V*x).sum()
    U2 = 0.9 * torch.log(x[0] +x[1] + 1) + 0.1 * torch.log(x[2]+1) + 0.2 * torch.log(x[0]+1) + 0.5 * torch.log(x[1]+1) + 0.3 * torch.log(x[2]+1) 
    U3 = (x[0]+x[1])/2 * 0.679
    R = s_var[-1] * B
    U = 10*U1+10*U2 +10*U3 + R 
    U.backward(torch.ones_like(U))
    _grad = s_var.grad
    s +=  1 * _grad
    s = torch.where(s < 0, 0, s)
    s /= s.sum()
print(U)
print(x)
print(s)
print('===')

z = z-x

for __ in range(1000):
    s_var = s.clone()
    s_var.requires_grad = True
    x = s_var[:-1]*B/p
    U1 = (V*x).sum()
    U2 = 0.3 * torch.log(x[0] +x[1] + 1) + 0.7 * torch.log(x[2]+1) + 0.1 * torch.log(x[0]+1) + 0.1 * torch.log(x[1]+1) + 0.2 * torch.log(x[2]+1) 
    U3 = (x[2]+x[1])/2 * 0.679
    R = s_var[-1] * B
    U = 10*U1+10*U2 +10*U3 + R 
    U.backward(torch.ones_like(U))
    _grad = s_var.grad
    s +=  1 * _grad
    s = torch.where(s < 0, 0, s)
    s /= s.sum()
print(U)
print(x)
print(s)
print('===')

z = z-x

print(z)