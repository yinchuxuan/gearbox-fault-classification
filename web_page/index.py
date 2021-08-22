from flask import Flask, request, abort, render_template
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
import time
from classification import Classifier
from classification import AutoEncoder_Filter
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

training_set_data = []

file = open('../data/training_set/showing_fft_data.txt')
for line in file.readlines():
    line_data = line.strip().replace("\n","")
    line_data = line_data.split()
    training_set_data.append(float(line_data[0]))

classifier = Classifier()

state_dict = torch.load("parameters.pth")

classifier.load_state_dict(state_dict)

def decode(tag):
    if tag == 0:
        return ("正常", "99.04%")

    if tag == 1:
        return ("缺齿", "97.12%")

    if tag == 2:
        return ("齿根断裂", "99.99%")

    if tag == 3:
        return ("齿面剥落", "99.99%")

    if tag == 4:
        return ("齿顶1级剥落", "99.99%")

    if tag == 5:
        return ("齿顶2级剥落", "98.08%")

    if tag == 6:
        return ("齿顶3级剥落", "97.12%")

    if tag == 7:
        return ("齿顶4级剥落", "99.99%")

    if tag == 8:
        return ("齿顶5级剥落", "99.04%")
    
app = Flask(__name__)

i = 0

@app.route('/')
def index():
    global i
    data = training_set_data[i : i + 360]
    tag = classifier.evaluate(torch.FloatTensor(data).view(-1, 1))
    state, accuracy = decode(tag)
    i = (i + 360) % len(training_set_data)
    return render_template('index.html', state = state, accuracy = accuracy, freqency_domain = data)

