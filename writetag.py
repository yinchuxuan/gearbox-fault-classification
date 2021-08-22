import numpy as np
import matplotlib.pyplot as plt

'''
training_data = open("data/training_set/data.txt", mode="a")

for i in range(1, 9):
    fileName = "data/spur_" + str(i)
    file = open(fileName + ".txt", mode="r")

    for j in range(0, 100000):
        line = file.readline()

        if not line:
            break
        
        training_data.write(line)

data = []

file = open('training_data_normal.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    data.append([float(line_data[0])])

training_data = np.array([])

training_tag = []

test_data = np.array([])

test_tag = []

for index in range(0, 3032640, 336960):
    if index < 2695680:
        training_data = np.append(training_data, data[index : index + 336960])
    else:
        test_data = np.append(test_data, data[index : index + 336960])

for i in range(104):
    for j in range(8):
        for k in range(9):
            training_tag.append(k)

for i in range(104):
    for k in range(9):
        test_tag.append(k)

np.savetxt("data/training_set/fft_data.txt", training_data)
np.savetxt("data/training_set/fft_tag.txt", np.array(training_tag))
np.savetxt("data/test_set/fft_data.txt", test_data)
np.savetxt("data/test_set/fft_tag.txt", np.array(test_tag))

data = []

for i in range(625):
    data.append(0)

np.savetxt("data/training_set/tmp_data.txt", np.array(data))
'''
training_set_data = []

file = open("data/training_set/fft_data.txt")
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    training_set_data.append([float(line_data[0])])

training_set_data = np.array(training_set_data)

showing_data = np.array([])

j = 0

for i in range(104):
    showing_data = np.append(showing_data, training_set_data[9 * 360 * i : 9 * 360 * i + 360])

    if i % 10 == 0:
        j = (j + 1) % 9
        showing_data = np.append(showing_data, training_set_data[9 * 360 * i + j * 360 : 9 * 360 * i + j * 360 + 360])

np.savetxt("data/training_set/showing_fft_data.txt", showing_data)
'''
accuracy = []

accuracy_low = []

accuracy_high = []

file = open('loss_freqency.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    accuracy.append([float(line_data[0])])

file = open('loss_72.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    accuracy_low.append([float(line_data[0])])

file = open('loss_3744.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    accuracy_high.append([float(line_data[0])])

y = np.arange(0, 1000, 1)
plt.plot(y, accuracy, label="batchsize=729")
plt.plot(y, accuracy_low, label="batchsize=72")
plt.plot(y, accuracy_high, label="batchsize=3744")
plt.legend()
plt.xlabel("epoch")
plt.title("loss function under different batch size")
plt.savefig("fig/result/loss_of_batch_size.png")
plt.close()
'''

