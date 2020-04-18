'''
python script to run Q-learning trading bot in Terminal for stock trajectory with given symbol

Implementation of neural network by pytorch

parser inputs:
"--symbol", "-s", help="set ticker symbol"
"--interval", "-i", help="set time step of data"
"--split", "-t", help="set date for train-test-splitting"


#example in terminal: python3 py_trader.py -s GOOGL -i daily -t '2018-01-01'
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
parser.add_argument("--split", "-t", help="set date for train-test-splitting")

args = parser.parse_args()

if args.symbol and args.split and args.interval:
    print("Load close price trajectory of %s" % args.symbol)

   # data = pd.read_csv('data/' + args.symbol + '.csv')
    data = loaddata(symbol = args.symbol, interval = args.interval)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    #data = data.drop('Unnamed: 0', axis = 1)
    #print(data.index.min(), data.index.max())
    #data.head()
    
    cut = args.split
    train = data[:cut]
    test = data[cut:]
    
    
env = Environment(train)

hidden_size=100
input_size=env.history_t+1
output_size=3
use_cuda  = False
lr = 0.001

Q = Q_Network(input_size, hidden_size, output_size)
Q_ast = copy.deepcopy(Q)

if use_cuda:
    Q = Q.cuda()
loss_function = nn.MSELoss()
optimizer = optim.Adam(list(Q.parameters()), lr=lr)

n_epochs = 10
step_max = len(env.data)-1
memory_size = 200
batch_size = 50

#obs, reward, done = env.step(5)

memory = []
total_step = 0
total_rewards = []
total_losses = []
epsilon = 1.0
epsilon_decrease = 1e-3
epsilon_min = 0.1
start_reduce_epsilon = 200
train_freq = 10
update_q_freq = 20
gamma = 0.97
show_log_freq = 5

print('Start Q-Learning')
start = time.time()
for epoch in range(n_epochs):
    print('epoch ' + str(epoch+1))
    pobs = env.reset()
    step = 0
    done = False
    total_reward = 0
    total_loss = 0

    while not done and step < step_max:

        # select act
        pact = np.random.randint(3)
        if np.random.rand() > epsilon:
            pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
            pact = np.argmax(pact.data)
            pact = pact.numpy()

        # act
        obs, reward, done = env.step(pact)

        # add memory
        memory.append((pobs, pact, reward, obs, done))
        if len(memory) > memory_size:
            memory.pop(0)

        # train or update q
        if len(memory) == memory_size:
            if total_step % train_freq == 0:
                shuffled_memory = np.random.permutation(memory)
                memory_idx = range(len(shuffled_memory))
                for i in memory_idx[::batch_size]:
                    batch = np.array(shuffled_memory[i:i+batch_size])
                    b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                    b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                    b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                    b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                    q = Q(torch.from_numpy(b_pobs))
                    q_ = Q_ast(torch.from_numpy(b_obs))
                    maxq = np.max(q_.data.numpy(),axis=1)
                    target = copy.deepcopy(q.data)
                    for j in range(batch_size):
                        target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                    Q.zero_grad()
                    loss = loss_function(q, target)
                    total_loss += loss.data.item()
                    loss.backward()
                    optimizer.step()
                    
            if total_step % update_q_freq == 0:
                Q_ast = copy.deepcopy(Q)
                
            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            #print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
torch.save(Q, 'model_' + str(args.symbol) + '.pth')
print('Successfully trained trading bot!')
            
            
print('Apply trained trader on test data...')            
test_env = Environment(test)
pobs = test_env.reset()
test_acts = []
test_rewards = []

for _ in range(len(test_env.data)-1):
    
    pact = Q(torch.from_numpy(np.array(pobs, dtype=np.float32).reshape(1, -1)))
    pact = np.argmax(pact.data)
    test_acts.append(pact.item())
            
    obs, reward, done = test_env.step(pact.numpy())
    test_rewards.append(reward)

    pobs = obs
        
test_profits = test_env.profits
print('test profits in USD: {:.2f}'.format(test_profits))
            