# py_RL_trader (in progress)

Python script to run Q-learning trading bot in terminal for stock trajectory with given symbol.
First train the model for symbol by choice with py_trader.py.

### parser inputs:
"--symbol", "-s", help="set ticker symbol" 

"--interval", "-i", help="set time step of data"

"--split", "-t", help="set date for train-test-splitting"


### example in terminal: 
python3 py_trader.py -s GOOGL -i daily -t '2018-01-01'


Implementation of neural network by pytorch. Inspired by the work of https://github.com/shivamakhauri04/TradingBot/

### required libraries
numpy, pandas, matplotlib, time, torch, copy, argparse, datetime, yfinance



