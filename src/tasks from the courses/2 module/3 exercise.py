# Задача
"""Перепишите пример, используя torch.optim.SGD"""

# Решение
import torch

w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)
alpha = 0.001
optimizer = torch.optim.SGD([w], lr=alpha)

for _ in range(500):
    function = (w + 7).log().log().prod()
    function.backward()
    optimizer.step()
    optimizer.zero_grad()

print(w)
