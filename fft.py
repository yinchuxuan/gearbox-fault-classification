import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import pandas as pd
import math

training_set_data = []

data = pd.read_csv("data.csv")

data = data.values

training_set_data = []

training_set_tag = []

test_set_data = []

test_set_tag = []

for i in range(104):
    for j in range(9):
        for k in range(9):
            training_set_data.append(data[j * 360 : j * 360 + 360, i * 9 + k])
            training_set_tag.insert(-1, k)
            x = np.arange(0, 360, 1)
            if i == 0:
                plt.plot(x, data[j * 360 : j * 360 + 360, i * 104 + k])
                plt.savefig("fig/time_domain/" + str(k) + "/" + str(j) + ".png")
                plt.close()

for i in range(104):
    for k in range(9):
        test_set_data.append(data[3600 -360 : 3600, i * 9 + k])
        test_set_tag.insert(-1, k)

np.savetxt("data/training_set/data.txt", np.resize(np.array(training_set_data), (-1, 1)))
np.savetxt("data/training_set/tag.txt", np.array(training_set_tag))
np.savetxt("data/test_set/data.txt", np.resize(np.array(test_set_data), (-1, 1)))
np.savetxt("data/test_set/tag.txt", np.array(test_set_tag))

'''
file = open('data/training_set/data.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    training_set_data.append([float(line_data[0]), float(line_data[1]), float(line_data[2])])

training_set_data = np.array(training_set_data)

fft_training_set_data = np.zeros((400000, 2))

for i in range(0, 104):
    for j in range(9):
            for k in range(9):
                t = training_set_data[k * 100000 + i * 1000 : k * 100000 + i * 1000 + 1000, j]
                fft_t = abs(fft(t))
                fft_t = fft_t[range(math.ceil(len(fft_t) / 2))]
                fft_t[0] = 0
                fft_training_set_data[k * 50000 + i * 500 : k * 50000 + i * 500 + 500, j] = fft_t
                x = np.arange(1, 500, 1)
                plt.plot(x.tolist(), (fft_t.tolist())[1:500])
                plt.savefig("data/fig/" + str(k) + "/" + str(j) + "_" + str(i) + ".png")
                plt.close()  

'''

#np.savetxt("data/training_set/fft_data.txt", fft_training_set_data)

#x = np.arange(0, 360, 1)

#plt.plot(x.tolist(), fft_t.tolist())

#plt.show()

