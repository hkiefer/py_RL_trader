import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
import yfinance as yf #have to be installed first!


#https://aroussi.com/post/python-yahoo-finance
    #https://github.com/ranaroussi/yfinance
def loaddata(symbol, interval, start_date = '1985-01-01' , verbose_plot = False): 
    if interval == 'daily':
        xlabel = 'days'
        trj = yf.Ticker(symbol)
        trj = trj.history(period='max', start = start_date, interval = '1d')
        
    elif interval == 'weekly':
        xlabel = 'weeks'
        trj = yf.Ticker(symbol)
        trj = trj.history(period='max', start = start_date, interval = '1wk')
        
    elif interval == 'hourly':
        xlabel = 'hours'
        trj = yf.Ticker(symbol)
        trj = trj.history(interval = '1h')
        
        
    elif interval == 'minutely':
        xlabel = 'minutes'
        trj = yf.Ticker(symbol)
        trj = trj.history(period="7d", interval = '1m')
    trj = trj.reset_index()
    
    trj = trj.fillna(method='ffill')  
    if interval != 'minutely':
        
        trj = trj.replace(to_replace=0, method='ffill')
    
    if verbose_plot:
        plt.plot(trj.index, trj['Close'], label = symbol)
        plt.xlabel(xlabel, fontsize = 'x-large')
        plt.ylabel("close", fontsize = 'x-large')
        plt.title('Loaded Trajectory')
        plt.tick_params(labelsize="x-large")
        plt.legend(loc = 'best')
        #plt.savefig("run_figures/trj.png", bbox_inches='tight')
        plt.show()
        plt.close()
        
    return trj

def load_csv(filename, start = '1980-01-01', value = 'Close', verbose_plot = False):
    
    trj = pd.read_csv(filename)
    
    start_date = start


    mask = (trj['Date'] > start_date)
    trj = trj.loc[mask]
    trj = trj.reset_index()

    if verbose_plot:
        plt.plot(trj.index,trj['Close'])
        plt.xlabel('t', fontsize = 'x-large')
        plt.ylabel("close", fontsize = 'x-large')
        plt.title('Loaded Trajectory')
        plt.tick_params(labelsize="x-large")
        plt.legend(loc = 'best')
        #plt.savefig("run_figures/trj.png", bbox_inches='tight')
        plt.show()
        plt.close()
    return trj #returns pandas Dataframe


def to_years(week, t0):
    return int(t0.year+(t0.isocalendar()[1]+week)/52.1429)

def week_range(n_weeks, t0):
    return int(to_years(np.arange(n_weeks), t0))

 
