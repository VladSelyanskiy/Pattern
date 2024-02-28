# Задание

"""
Обучим нейронную сеть для задачи регрессии
Возьмем более сложную функцию в качестве таргета
Кроме того, мы хотим получить хорошую метрику MAE на валидации 

Получите метрику не хуже 0.03
"""

# Решение

import torch


def target_function(x):
    return 2**x * torch.sin(2**-x)


class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


net = RegressionNet(40)

# ------Dataset preparation start--------:
x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20.0
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)
# ------Dataset preparation end--------:


optimizer = torch.optim.Adam(
    net.parameters(), lr=0.1
)  # Можно брать либо lr = 0.1, либо lr = 0.01


def loss(pred, target):
    squares = abs(pred - target)
    return squares.mean()


for epoch_index in range(400):
    optimizer.zero_grad()

    y_pred = net.forward(x_train)
    loss_value = loss(y_pred, y_train)

    loss_value.backward()

    optimizer.step()


# Проверка осуществляется вызовом кода:
def metric(pred, target):
    return (pred - target).abs().mean()


print(metric(net.forward(x_validation), y_validation).item())
# (раскомментируйте, если решаете задание локально)
