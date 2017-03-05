# coding=utf-8
# Задача 1. Линейно неразделимая классификация с 2 классами
# Создайте и обучите ИНС, которая решает следующую задачу классификации:
# I класс: {(x1,x2), x2 > x12-4x1+5; 0 ≤ x1 ≤ 5; 0 ≤ x2 ≤ 5; x1R; x2R}
# II класс: {(x1,x2), x2 < x12-4x1+5; 0 ≤ x1 ≤ 5; 0 ≤ x2 ≤ 5; x1R; x2R}

import numpy as np
import pylab as pl
import neurolab as nl

x = np.random.uniform(0, 5, (150, 2))  # 150 двумерных точек от 0 до 5
# определяем классы по условию x2 ? x12-4x1+5
y = np.sign(x[:, 1] - (x[:, 0] ** 2 - x[:, 0] * 4 + 5)).reshape(150, 1)
pl.plot(x[:, 0], (x[:, 0] ** 2 - x[:, 0] * 4 + 5), 'b*')

# Размер и min max пределы входного слоя, количество нейронов на скрытом и выходном слое
net = nl.net.newff([[0, 5], [0, 5]], [5, 1])

err = net.train(x, y, show=10)
a = net.sim(x)
for i in range(len(y)):
    if a[i] > 0:
        pl.plot(x[i, 0], x[i, 1], 'ro')
    if a[i] < 0:
        pl.plot(x[i, 0], x[i, 1], 'go')

xx, yy = np.meshgrid(np.arange(0, 5, 0.2), np.arange(0, 5, 0.2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
t = np.concatenate((xx, yy), axis=1)
a = net.sim(t)
gr1 = t[a[:, 0] > 0]
gr2 = t[a[:, 0] < 0]

pl.plot(gr1[:, 0], gr1[:, 1], 'r+', gr2[:, 0], gr2[:, 1], 'g+')
pl.axis([0, 5, 0, 5])
pl.show()
