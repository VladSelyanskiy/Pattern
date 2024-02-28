# Задача
"""Реализуйте градиентный спуск для той же функции
Пусть начальным приближением будет w = [[5,10],[1,2]], шаг градиентного спуска alpha=0.001"""

# Решение
import torch

w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)
alpha = 0.001

for _ in range(500):
    function = (w + 7).log().log().prod()
    function.backward()
    w.data -= alpha * w.grad
    w.grad.zero_()

# print(w)
