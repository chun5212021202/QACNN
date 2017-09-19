
from model import MODEL
from utility import utility
import time
import json
import numpy as np
import random
import os
import sys


EPOCH = 50
X_DIMENSION = 300
LEARNING_RATE = 0.001
BATCHSIZE = 20
DROPOUT = 0.8
choice = 5
max_plot_num = 101
max_len = [100,50,50]
parameter_size = {
	'cnn_filterSize':{'filter1':[1,3,5],'filter2':[1,3,5]},	
	'cnn_filterNum':128,
	'cnn_filterNum2':128,
	'dnn_hiddenUnits':128
}

CNN_FILTER_SIZE = parameter_size['cnn_filterSize']['filter1']

CNN_FILTER_SIZE2 = parameter_size['cnn_filterSize']['filter2']
CNN_FILTER_NUM = int(parameter_size['cnn_filterNum'])
CNN_FILTER_NUM2 = int(parameter_size['cnn_filterNum2'])
DNN_WIDTH = int(parameter_size['dnn_hiddenUnits'])

plotFilePath = "output_data/plot/"
parameterPath = 'parameter/'

print('###############################################################')
print('Epoch             :',EPOCH)
print('X Dimension       :', X_DIMENSION)
print('Learning Rate     :', LEARNING_RATE)
print('Drop Out          :', DROPOUT)
print('Plot Sentence Lum :', max_plot_num)
print('Plot Sentence Len :', max_len[0])
print('Q Sentence Len    :', max_len[1])
print('Ans Sentence Len  :', max_len[2])
print('CNN Filter Size   :', CNN_FILTER_SIZE)
print('CNN Filter Size2   :', CNN_FILTER_SIZE2)
print('CNN Filter Num    :', CNN_FILTER_NUM)
print('DNN Output Size   :', DNN_WIDTH,'->',1)
print('###############################################################','\n')


with open('output_data/question/qa.train.json') as data_file:    
	train_q = json.load(data_file)
with open('output_data/question/qa.val.json') as data_file:
	val_q = json.load(data_file)

start = time.time()

acm_net = MODEL.MODEL(BATCHSIZE,X_DIMENSION,DNN_WIDTH,CNN_FILTER_SIZE,CNN_FILTER_SIZE2,CNN_FILTER_NUM,CNN_FILTER_NUM2,LEARNING_RATE,DROPOUT,choice,max_plot_num,max_len,parameterPath)
acm_net.initialize()
accuracy = utility.train(train_q,val_q,plotFilePath,acm_net,EPOCH,LEARNING_RATE,BATCHSIZE,DROPOUT,choice,max_len)
accuracy_train = np.array(accuracy['train'])
accuracy_val = np.array(accuracy['val'])
	
print('val: ', accuracy_val, np.amax(accuracy_val))
print('train: ',accuracy_train, np.amax(accuracy_train))

eval_time = time.time()-start
print('use time: ',eval_time)

