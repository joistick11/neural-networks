# coding=utf-8
# Задача 3. Линейно разделимая классификация с 4 классами (персептрон)
# Создайте и обучите ИНС, которая решает следующую задачу классификации:
# I класс: {(x1,x2), x2 < 5-x1, x2 < x1+1; 0 ≤ x1 ≤ 5; 0 ≤ x2 ≤ 5; x1R; x2R}
# II класс: {(x1,x2), x2 < 5-x1, x2 > x1+1; 0 ≤ x1 ≤ 5; 0 ≤ x2 ≤ 5; x1R; x2R}
# III класс: {(x1,x2), x2 > 5-x1, x2 < x1+1; 0 ≤ x1 ≤ 5; 0 ≤ x2 ≤ 5; x1R; x2R}
# IV класс: {(x1,x2), x2 > 5-x1, x2 > x1+1; 0 ≤ x1 ≤ 5; 0 ≤ x2 ≤ 5; x1R; x2R}

import numpy as np
import neurolab as nl
import pylab as pl


def determine_class(x):
    def class_1(x):
        return x[i, 1] < 5 - x[i, 0] and x[i, 1] < 1 + x[i, 0]

    def class_2(x):
        return x[i, 1] < 5 - x[i, 0] and x[i, 1] > 1 + x[i, 0]

    def class_3(x):
        return x[i, 1] > 5 - x[i, 0] and x[i, 1] < 1 + x[i, 0]

    def class_4(x):
        return x[i, 1] > 5 - x[i, 0] and x[i, 1] > 1 + x[i, 0]

    y_result = []
    for i in range(len(x)):
        if class_1(x):
            y_result.append([0, 0])
        elif class_2(x):
            y_result.append([0, 1])
        elif class_3(x):
            y_result.append([1, 0])
        elif class_4(x):
            y_result.append([1, 1])
    return y_result


def determine_color(p):
    if list(p) == [0., 0.]:
        return 'ro'
    if list(p) == [0., 1.]:
        return 'bo'
    if list(p) == [1., 0.]:
        return 'yo'
    if list(p) == [1., 1.]:
        return 'go'

# Создаем выборку на 150 элементов
x = np.random.uniform(0, 5, (150, 2))
# И определяем классы получившихся элементов
y = determine_class(x)

# Создаем персептрон с 2 нейронами на входе и 2 нейронами в выходном слое
net = nl.net.newp([[0, 5], [0, 5]], 2)

error = net.train(x, y, epochs=100, show=10, lr=0.1)
a = net.sim(x)

# Рисуем разделяющие гиперплоскости персептрона
w = net.layers[0].np['w'][0]
b = net.layers[0].np['b'][0]
w_1 = net.layers[0].np['w'][1]
b_1 = net.layers[0].np['b'][1]
pl.plot([0., 5.], [-b / w[1], (-b - 5 * w[0]) / w[1]], 'c')
pl.plot([0., 5.], [-b_1 / w_1[1], (-b_1 - 5 * w_1[0]) / w_1[1]], 'y')

for i in range(len(x)):
    pl.plot(x[i, 0], x[i, 1], determine_color(a[i]))

pl.axis([0, 5, 0, 5])
pl.show()
