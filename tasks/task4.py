# coding=utf-8
# Задача 4. Кластеризация (сеть Кохонена)
# Создайте и обучите ИНС, которая находит центры трех кластеров множества точек 2-мерного пространства:
# {(0, 7), (4.7, 9.7), (4.6, 4.6), (1.3, 7.7), (4.4, 9.7), (4.6, 3.6), (0.4, 7.5), (4.3, 9.9),
# (5.6, 3.9), (0.2, 7), (4.8, 8.1), (5.1, 4.3), (0.3, 8), (4.3, 9.3), (4, 3.3), (0, 7.6), (4.2, 8.6),
# (5.2, 3.8), (0.1, 7.6), (4.2, 8.6), (5.2, 3.8), (0.1, 7.5), (4.3, 9.5), (4.5, 3.3)}

import numpy as np
import neurolab as nl
import pylab as pl

x = np.array([[0, 7], [4.7, 9.7], [4.6, 4.6], [1.3, 7.7],
              [4.4, 9.7], [4.6, 3.6], [0.4, 7.5], [4.3, 9.9],
              [5.6, 3.9], [0.2, 7], [4.8, 8.1], [5.1, 4.3],
              [0.3, 8], [4.3, 9.3], [4, 3.3], [0, 7.6],
              [4.2, 8.6], [5.2, 3.8], [0.1, 7.6], [4.2, 8.6],
              [5.2, 3.8], [0.1, 7.5], [4.3, 9.5], [4.5, 3.3]])
np.random.shuffle(x)

# Создание и обучение слоя Кохонена два входных и три выходных нейрона
net = nl.net.newc([[0.0, 9.9], [0.0, 9.9]], 3)
error = net.train(x, epochs=400, show=60)

# Рисование кластеров и их центров
w = net.layers[0].np['w']
pl.plot(x[:, 0], x[:, 1], 'g+', w[:, 0], w[:, 1], 'ro')
pl.legend(['train samples', 'train centers'], bbox_to_anchor=(0.8, 0.7))
pl.show()
