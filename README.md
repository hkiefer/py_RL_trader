python script to run Q-learning trading bot in Terminal for stock data set in subfolder /data by choice

Implementation of neural network by pytorch

### required libraries
numpy, pandas, matplotlib, time, torch, copy, argparse


### parser inputs:
"--symbol", "-s", help="set ticker symbol"
"--split", "-t", help="set date for train-test-splitting"


### example in terminal: 
python3 py_trader.py -s googl -t '2016-01-01'
