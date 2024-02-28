# Задача
""" Реализуйте расчет градиента для функции"""

# Решение
import torch

w = torch.tensor([[5.0, 10.0], [1.0, 2.0]], requires_grad=True)

function = torch.log(torch.log(w + 7)).prod()
function.backward()
