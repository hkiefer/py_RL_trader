'''
parser inputs:
"--symbol", "-s", help="set ticker symbol"
"--interval", "-i", help="set time step of data"



#example in terminal: python3 trade_advice.py -s GOOGL -i daily 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import copy
import argparse

from loaddata import *
from q_learner import *


    
    
#--------------------code-----------------

parser =  argparse.ArgumentParser()
parser.add_argument("--symbol", "-s", help="set ticker symbol")
parser.add_argument("--interval", "-i", help="set time step of data")

args = parser.parse_args()

if args.symbol and args.interval:
    print("Load close price trajectory of %s" % args.symbol)

   # data = pd.read_csv('data/' + args.symbol + '.csv')
    data = loaddata(symbol = args.symbol, interval = args.interval)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    #data = data.drop('Unnamed: 0', axis = 1)
    #print(data.index.min(), data.index.max())
    #data.head()
    
    Q = torch.load('model_' + str(args.symbol) + '.pth')
    Q.eval()




    test_env = Environment(data[-30:])
    pobs = test_env.reset()

    pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
    pact = np.argmax(pact.data)
    #print(pact.detach().numpy())
    # act = 0: stay, 1: buy, 2: sell
    if pact.detach().numpy() == 0:
        print('hold')
        
    elif pact.detach().numpy() == 1:
        print('buy')

    else:
        print('sell') 