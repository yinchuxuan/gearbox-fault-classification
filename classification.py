#!/usr/bin/python
# coding=utf-8

import numpy as np
import torch.nn as nn
from torch.nn import functional
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence,pack_padded_sequence
import torch
import math
import re
import matplotlib.pyplot as plt
import collections

dim = 1

seq_length = 360

batch_size = 9 * 81

embedding_size = 16

fc_num = 128

classification_type = 9

hidden_size = 32

lr = 1e-4

epoch = 1500

# load data
training_set_data = []

test_set_data = []


file = open('data/training_set/generator_training_unbalance.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    training_set_data.append([float(line_data[0])])

training_set_data = np.array(training_set_data)
training_set_tag = np.loadtxt('data/training_set/generator_training_unbalance_tag.txt')
training_set_tag = training_set_tag.astype(np.int)

file = open('data/test_set/fft_data.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    test_set_data.append([float(line_data[0])])

test_set_data = np.array(test_set_data)
test_set_tag = np.loadtxt('data/test_set/fft_tag.txt')
test_set_tag = test_set_tag.astype(np.int)


def get_batches(data, tag):
    '''Create a generator of batches as a tuple (inputs, targets)'''

    for index in range(0, len(data), batch_size * seq_length):
        if index + batch_size * seq_length < len(data):
            x = torch.FloatTensor(data[index : index + batch_size * seq_length])
            x = x.view(-1, seq_length, dim)
            y = torch.Tensor([tag[int(index / seq_length): int(index / seq_length) + batch_size]])

            yield x, y
        else:
            break

# define model

# AutoEncoder
class AutoEncoder_Filter(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(    # input size = 512 * dim * 512
            nn.Conv1d(1, 16, 4),    # output size = 512 * 16 * 509
            nn.ReLU(),
            #nn.MaxPool1d(dim, 2),     # output size = 512 * 16 * 254
            nn.Conv1d(16, 64, 4),   # output size = 512 * 64 * 251
            nn.ReLU(),
            #nn.MaxPool1d(dim, 2),     # output size = 512 * 64 * 125
            nn.ReLU(),
            nn.Conv1d(64, 256, 4),  # output size = 512 * 256 * 122
            nn.ReLU(),
            #nn.MaxPool1d(4, 2)      # output size = 512 * 256 * 120
        )

        #Decoder
        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(4, 2, 0),    # output size = 512 * 256 * 122
            nn.ReLU(),
            nn.ConvTranspose1d(256, 64, 4),             # output size = 512 * 64 * 125
            nn.ReLU(),
            #nn.MaxUnpool1d(dim, 2, 0),    # output size = 512 * 64 * 251
            nn.ReLU(),
            nn.ConvTranspose1d(64, 16, 4),              # output size = 512 * 16 * 254
            nn.ReLU(),
            #nn.MaxUnpool1d(dim, 2, 0),    # output size = 512 * 16 * 509
            nn.ReLU(),
            nn.ConvTranspose1d(16, dim, 4)                # output size = 512 * dim * 512
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder.forward(x)
        return (self.decoder.forward(x)).permute(0, 2, 1)

# This is unneccessary, right?
# class CNN_Feature_capture(nn.Module):

class Self_Attention_header(nn.Module):
    def __init__(self):
        super().__init__()
        self.Query = nn.Linear(dim, embedding_size)
        self.Key = nn.Linear(dim, embedding_size)
        self.Value = nn.Linear(dim, embedding_size)

    def forward(self, input_vector):
        list = [(self.Query(v), self.Key(v), self.Value(v)) for v in input_vector]
        result = torch.zeros(len(list), embedding_size)

        for i in range(len(list)):
            (Q1, K1, V1) = list[i]
            score = torch.zeros(len(list))

            for i in range(len(list)):
                (Q2, K2, V2) = list[i]
                score[i] = Q1.dot(K2) / math.sqrt(embedding_size)

            p = functional.softmax(score, dim = 0)
            value = torch.zeros(embedding_size)

            for i in range(len(list)):
                (_, _, V) = list[i]
                value = value + p[i] * V

            result[i, :] = value

        return result

# Self_Attention based Encoder
class Self_Attention_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.header1 = Self_Attention_header()
        #self.header2 = Self_Attention_header()
        #self.headerdim = Self_Attention_header()
        #self.header4 = Self_Attention_header()
        self.W = nn.Linear(embedding_size, dim)
        self.ln = nn.LayerNorm(dim)

        self.fc = nn.Sequential(
            nn.Linear(dim, fc_num),
            nn.Sigmoid(),
            nn.Linear(fc_num, dim)
        )

    def forward(self, input_vector):
        #v1, v2, vdim, v4 = self.header1.forward(input_vector), self.header2.forward(input_vector), self.headerdim.forward(input_vector), self.header4.forward(input_vector)
        #v = torch.cat((v1, v2, vdim, v4), 1)
        v = self.header1.forward(input_vector)
        z = self.W(v)
        z = (z + input_vector)

        for i in range(z.shape[0]):
            z[i,:] = self.ln(z[i,:])

        result = self.fc(z)
        result = (result + z)

        for i in range(result.shape[0]):
            result[i,:] = self.ln(result[i,:])
        
        return result

# Classifier
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        #self.filter1 = AutoEncoder_Filter()
        #self.encoder1 = Self_Attention_Encoder()
        #self.encoder2 = Self_Attention_Encoder()
        #self.encoderdim = Self_Attention_Encoder()
        #self.encoder4 = Self_Attention_Encoder()
        '''
        self.extraction = nn.Sequential(
            nn.Conv1d(dim, 4, 22),     # output_size = 1 * 4 * 480
            nn.ReLU(),
            nn.MaxPool1d(3, 3),    # output_size = 1 * 4 * 160
            nn.Conv1d(4, 8, 11),    # output_size = 1 * 8 * 150
            nn.ReLU(),
            nn.MaxPool1d(3, 3),    # output_size = 1 * 8 * 50
            nn.Conv1d(8, 16, 6),   # output_size = 1 * 16 * 45
            nn.ReLU(),
            nn.MaxPool1d(3, 3),    # output_size = 1 * 16 * 14      
        )
        '''

        self.extraction = nn.Sequential(collections.OrderedDict([
            ("conv1", nn.Conv1d(dim, 4, 12)),      # output_size = 1 * 4 * 350
            ("ReLu1", nn.ReLU()),
            ("MaxPool1", nn.MaxPool1d(5, 5)),         # output_size = 1 * 4 * 70
            ("conv2", nn.Conv1d(4, 8, 7)),         # output_size = 1 * 8 * 64
            ("ReLu2", nn.ReLU()),
            ("MaxPool2", nn.MaxPool1d(4, 4)),         # output_size = 1 * 8 * 16
            ("conv3", nn.Conv1d(8, 16, 2)),        # output_size = 1 * 16 * 15
            ("MaxPool3", nn.MaxPool1d(3, 3))          # output_size = 1 * 16 * 4 
        ]))

        self.lstm = nn.LSTM(dim, 256, 2)

        self.fc = nn.Sequential(
            nn.Linear(16 * 4, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, classification_type)
        )

    def forward(self, input_vector):
        #batch_size = input_vector.shape[0]
        #tag = torch.zeros(batch_size, classification_type)
        #input_vector = self.filter1.forward(input_vector)
        

        #x = self.encoder1.forward(filter_vector[i,:])
        #x = self.encoder2.forward(x)
        #x = self.encoderdim.forward(x)
        #x = self.encoder4.forward(x)
        input_vector = input_vector.permute(0, 2, 1)
        x = self.extraction(input_vector)
        x = x.view(batch_size, -1)
        return functional.softmax(self.fc(x), dim = 1)

    def evaluate(self, input_vector):
        #input_vector = input_vector.view(1, seq_length, dim)
        #input_vector, _ = self.lstm(input_vector)
        #input_vector = input_vector.view(seq_length, 256)
        #input_vector = self.filter1.forward(input_vector)
        #input_vector = input_vector.view(seq_length, dim)
        #result = torch.zeros(seq_length)
        #tag = torch.zeros(seq_length)

        #x = encoder1.forward(input_vector)
        #x = encoder2.forward(x)
        #x = encoderdim.forward(x)
        #x = encoder4.forward(x)
        #x = self.fc(input_vector)
        #tag = torch.argmax(functional.softmax(x, dim = 1), dim = 1)

        input_vector = input_vector.permute(1, 0)
        x = input_vector.view(1, dim, seq_length)
        x = self.extraction(x)
        x = x.view(1, -1)
        x = self.fc(x)
        x = x.view(-1)

        return torch.argmax(functional.softmax(x, dim = 0), dim = 0) 

# instantiate model
classifier_model = Classifier()
optimizer = optim.Adam(classifier_model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
lossArray = []

# train model
def train(epoch):
    loss = 0

    for i in range(epoch):
        for sources, targets in get_batches(training_set_data, training_set_tag):
            optimizer.zero_grad()
            outputs = classifier_model.forward(sources)
            tag = torch.tensor(targets).long()
            loss = criterion(outputs.view(-1, classification_type), tag.view(-1))   
            loss.backward()
            optimizer.step()

        print(loss)
        lossArray.append(loss)

        #if i == epoch - 1:
            #test2()
        #elif i % 100 == 0:
            #test1()



Accuracy = []

resultList = [0, 0, 0, 0, 0, 0, 0, 0, 0]

listarray = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],]

correct = 0

datalength = 0

def test1():
    global datalength
    global correct
    count = 0
    data_length = 0
    datalength = 0
    correct = 0

    for index in range(0, len(test_set_data), seq_length):
        if index + seq_length < len(test_set_data):
            x = torch.FloatTensor(test_set_data[index : index + seq_length])
            x.view(seq_length, dim)
            #y = torch.FloatTensor(test_set_tag[index : index + seq_length])
            y = test_set_tag[int(index / seq_length)] 

            prediction = classifier_model.evaluate(input_vector=x)

            if prediction == y:
                correct = correct + 1
                #resultList[y] = resultList[y] + (1 / 104)

            #listarray[y][prediction] = listarray[y][prediction] + (1 / 104)
            datalength = datalength + 1

    Accuracy.append(float(correct) / float(datalength))

def test2():
    global datalength
    global correct
    count = 0
    data_length = 0

    for index in range(0, len(test_set_data), seq_length):
        if index + seq_length < len(test_set_data):
            x = torch.FloatTensor(test_set_data[index : index + seq_length])
            x.view(seq_length, dim)
            #y = torch.FloatTensor(test_set_tag[index : index + seq_length])
            y = test_set_tag[int(index / seq_length)] 

            prediction = classifier_model.evaluate(input_vector=x)

            if prediction == y:
                correct = correct + 1
                resultList[y] = resultList[y] + (1 / 104)

            listarray[y][prediction] = listarray[y][prediction] + (1 / 104)
            datalength = datalength + 1

    Accuracy.append(correct / datalength)

train(epoch)

torch.save(classifier_model.state_dict(), "g_parameters.pth")


'''
cnn_weight_1 = classifier_model.extraction.conv1.weight.data

cnn_weight_1 = (cnn_weight_1.view(-1)).numpy().tolist()

cnn_weight_2 = classifier_model.extraction.conv2.weight.data

cnn_weight_2 = cnn_weight_2.view(-1).numpy().tolist()

cnn_weight_3 = classifier_model.extraction.conv3.weight.data

cnn_weight_3 = cnn_weight_3.view(-1).numpy().tolist()

grey1 = []

grey2 = []

grey3 = []

for i in range(50):
    grey1.append(cnn_weight_1)
    grey2.append(cnn_weight_2)
    grey3.append(cnn_weight_3)

plt.imshow(grey1, cmap=plt.cm.gray)

plt.title("kernel1")

plt.savefig("fig/result/kernel1.png")

plt.close()

plt.imshow(grey2, cmap=plt.cm.gray)

plt.title("kernel2")

plt.savefig("fig/result/kernel2.png")

plt.close()

plt.imshow(grey3, cmap=plt.cm.gray)

plt.title("kernel3")

plt.savefig("fig/result/kernel3.png")

plt.close()

lossArray = np.array(lossArray)

x = np.arange(0, lossArray.size, 1)

plt.plot(x, lossArray)

plt.title("Loss function curve")

plt.xlabel("epoch")

plt.savefig("fig/result/lossfig.png")

plt.close()

x = np.arange(0, 11, 1)

plt.plot(x, Accuracy)

plt.title("Accuracy curve")

plt.xlabel("epoch")

plt.ylabel("accuracy")

plt.savefig("fig/result/accuracy.png")

np.savetxt("accuracy_3744.txt", np.array(Accuracy))

np.savetxt("loss_3744.txt", np.array(lossArray))

plt.close()

y = np.arange(0, 9, 1)

plt.bar(y, resultList, 0.5)

plt.title("Classification accuracy on different working condition")

plt.xlabel("working condition")

plt.ylabel("accuracy")

for a, b in zip(x, resultList):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)

plt.savefig("fig/result/resultAll.png")

plt.close()

for i in range(9):
    plt.bar(y, listarray[i], 0.5)

    plt.title("Prediction on tag " + str(i) + " working condition")

    plt.xlabel("working condition")

    plt.ylabel("possibility")

    for a, b in zip(y, listarray[i]):
        plt.text(a, b + 0.01, '%.4f' % b, ha='center', va= 'bottom',fontsize=9)

    plt.savefig("fig/result/result" + str(i) + ".png")

    plt.close()
'''

