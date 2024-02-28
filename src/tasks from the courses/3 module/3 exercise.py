# Задание

"""
В этом задании вам нужно подобрать seed, чтобы функция random.randint(0, 10) выдала число 5
"""

# Решение

import random

i, required = -1, -1

while True:
    i += 1
    random.seed(i)
    if random.randint(0, 10) == 5:
        required = i
        break

random.seed(required)

print(random.randint(0, 10))
