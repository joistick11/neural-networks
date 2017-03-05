# coding=utf-8
# Задача 2. Нелинейная аппроксимация (многослойный персептрон)
# Создайте и обучите ИНС, которая решает задачу аппроксимации функции
# y = sin √x на множестве 0 ≤ x ≤ 5.

import neurolab as nl
import numpy as np
import pylab as pl

# Генерируем 200 точек
x = np.linspace(0, 5, 200)
y = np.sin(np.sqrt(x))

net_input = x.reshape(len(x), 1)
net_target = y.reshape(len(x), 1)

# Создаем сеть с двумя слоями
net = nl.net.newff([[0, 5]], [10, 1])

# Тренируем сеть
error = net.train(net_input, net_target, epochs=500, show=100, goal=0.02)

# Тестируем
out = net.sim(net_input)

pl.subplot(211)
pl.plot(error)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')

x2 = np.linspace(0.0, 5.0, 200)
y2 = net.sim(x2.reshape(x2.size, 1)).reshape(x2.size)

pl.subplot(212)
pl.plot(x2, y2, '-', x, y, '.')
pl.legend(['train target', 'net output'])
pl.show()
