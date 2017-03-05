# coding=utf-8
# Задача 5. Линейно неразделимая классификация с 3 классами (сеть встречного распространения)
# Создайте и обучите ИНС, которая решает следующую задачу классификации:
# I класс: {(x1,x2), (x1-1)2+(x2-1)2 < 1; -1 ≤ x1 ≤ 5; -1 ≤ x2 ≤ 5; x1R; x2R}
# II класс: {(x1,x2), (x1-3)2+(x2-3)2 < 1; -1 ≤ x1 ≤ 5; -1 ≤ x2 ≤ 5; x1R; x2R}
# III класс: {(x1,x2), (x1-1)2+(x2-1)2 > 1; (x1-3)2+(x2-3)2 > 1; -1 ≤ x1 ≤ 5;
# -1 ≤ x2 ≤ 5; x1R; x2R}

import numpy as np
import neurolab as nl
import pylab as pl


def determine_class(x):
    def class_1(x):
        return (x[i, 0] - 1) ** 2 + (x[i, 1] - 1) ** 2 < 1

    def class_2(x):
        return (x[i, 0] - 3) ** 2 + (x[i, 1] - 3) ** 2 < 1

    def class_3(x):
        return (x[i, 0] - 1) ** 2 + (x[i, 1] - 1) ** 2 > 1

    y_result = np.zeros((200, 3))
    for i in range(len(x)):
        if class_1(x):
            y_result[i, 0] = 1
        elif class_2(x):
            y_result[i, 1] = 1
        elif class_3(x):
            y_result[i, 2] = 1
    return y_result


# Генерация 200 двухмерных точек, разделение их на классы
x = np.random.uniform(-1, 5, (200, 2))
y = determine_class(x)
class1 = x[y[:, 0] > 0]
class2 = x[y[:, 1] > 0]
class3 = x[y[:, 2] > 0]

# Рисование точек по классам
pl.plot(class1[:, 0], class1[:, 1], 'ro', class2[:, 0], class2[:, 1], 'yo', class3[:, 0], class3[:, 1], 'bo')

# Создание и обучение сети встречного распространения
net = nl.net.newlvq(nl.tool.minmax(x), 20, [.2, .4, .4])
error = net.train(x, y, epochs=1000, goal=-1)

# Опрос обученной сети
xx, yy = np.meshgrid(np.arange(-1, 5, 0.2), np.arange(-1, 5, 0.2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
t = np.concatenate((xx, yy), axis=1)
a = net.sim(t)

# Визуализация результатов
gr1 = t[a[:, 0] > 0]
gr2 = t[a[:, 1] > 0]
gr3 = t[a[:, 2] > 0]
pl.plot(gr1[:, 0], gr1[:, 1], 'r+', gr2[:, 0], gr2[:, 1], 'y+', gr3[:, 0], gr3[:, 1], 'b+')
# pl.axis([-3.2, 3.2, -3, 3])
# pl.legend(['class 1', 'class 2','class 3', 'detected class 1', 'detected class 2', 'detected class 3'])
pl.show()
