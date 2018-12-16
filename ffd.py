import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpEngine import mpJobList
import statsmodels.api as sm

def getWeights(d, size):
    # thres>0 drops insignificant weights
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        # w = np.append(w, w_) # duno why w_ or w or something turns to np.ndarray suddenly, should be a list somewhat, may give a bug if d is not a np. float
        w.append(w_)
    w = np.array(w[ : : -1]).reshape(-1, 1)
    return w
#———————————————————————————————————————-
def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size = size)
        w_ = pd.DataFrame(w_, index = range(w_.shape[0])[ : : -1], columns = [d])
        w = w.join(w_, how = 'outer')
    ax = w.plot()
    ax.legend(loc = 'upper left')
    plt.show()
    return
#———————————————————————————————————————-
if __name__ == '__main__':
    plotWeights(dRange = [0, 1], nPlots = 11, size = 6)
    plotWeights(dRange = [1, 2], nPlots = 11, size = 6)

def fracDiff(series, d, thres = .01):
    """
    Increasing width window, with treatment of NaNs (Standard Fracdiff, expanding window)
    Note 1: For thres=1, nothing is skipped.
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    #1) Compute weights for the longest series
    w = getWeights(d, series.shape[0]) # each obs has a weight
    #2) Determine initial calcs to be skipped based on weight-loss threshold
    w_ = np.cumsum(abs(w)) # cumulative weights
    w_ /= w_[-1] # determine the relative weight-loss
    skip = w_[w_ > thres].shape[0]  # the no. of results where the weight-loss is beyond the acceptable value
    #3) Apply weights to values
    df = {} # empty dictionary
    for name in series.columns:
        # fill the na prices
        seriesF = series[[name]].fillna(method = 'ffill').dropna()
        df_ = pd.Series() # create a pd series
        for iloc in range(skip, seriesF.shape[0]):
            loc = seriesF.index[iloc] # find the iloc th obs 
            
            test_val = series.loc[loc,name] # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            
            if not np.isfinite(test_val).any():
                 continue # exclude NAs
            try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
                df_.loc[loc]=np.dot(w[-(iloc+1):,:].T, seriesF.loc[:loc])[0,0]
            except:
                continue
        df[name] = df_.copy(deep = True)
    df = pd.concat(df, axis = 1)
    return df

def getWeights_FFD(d, thres):
    # thres>0 drops insignificant weights
    w = [1.]
    k = 1
    while abs(w[-1]) >= thres:  
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
        k += 1
    w = np.array(w[ : : -1]).reshape(-1, 1)[1 : ]  
    return w

def fracDiff_FFD(series, d, thres = 1e-5):
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    #1) Compute weights for the longest series
    w = getWeights_FFD(d, thres)
    # w = getWeights(d, series.shape[0])
    #w=getWeights_FFD(d,thres)
    width = len(w) - 1
    #2) Apply weights to values
    df = {} # empty dict
    for name in series.columns:
        seriesF = series[[name]].fillna(method = 'ffill').dropna()
        df_ = pd.Series() # empty pd.series
        for iloc1 in range(width, seriesF.shape[0]):
            loc0 = seriesF.index[iloc1 - width]
            loc1 = seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):
                continue # exclude NAs
            #try: # the (iloc)^th obs will use all the weights from the start to the (iloc)^th
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0 : loc1])[0, 0]
            # except:
            #     continue
            
        df[name] = df_.copy(deep = True)
    df = pd.concat(df, axis = 1)
    return df

def plotMinFFD():
    from statsmodels.tsa.stattools import adfuller
    path = './'
    instName ='ES1_Index_Method12'
    out = pd.DataFrame(columns= ['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])
    df0 = pd.read_csv(path + instName +'.csv',index_col = 0, parse_dates = True)
    for d in np.linspace(0, 1, 11):
        df1 = np.log(df0[['Close']]).resample('1D').last() # downcast to daily obs
        df2 = fracDiff_FFD(df1, d, thres = .01)
        corr = np.corrcoef(df1.loc[df2.index, 'Close'], df2['Close'])[0, 1]
        df2 = adfuller(df2['Close'], maxlag = 1, regression = 'c', autolag = None)
        out.loc[d] = list(df2[ : 4]) + [df2[4]['5%']] + [corr] # with critical value
    out.to_csv(path + instName + '_testMinFFD.csv')
    out[['adfStat', 'corr']].plot(secondary_y = 'adfStat')
    plt.axhline(out['95% conf'].mean(), linewidth = 1, color = 'r', linestyle = 'dotted')
    plt.savefig(path + instName + '_testMinFFD.png')
    return

def get_optimal_ffd(data, start = 0, end = 1, interval = 10, t=1e-5):
    
    d = np.linspace(start,end,interval)
    out = mpJobList(mp_get_optimal_ffd, ('molecules', d), redux = pd.DataFrame.append, data = data)

    return out


def mp_get_optimal_ffd(data, molecules, t = 1e-5):
    
    cols = ['adfStat','pVal','lags','nObs','95% conf']
    out = pd.DataFrame(columns=cols)
    
    for d in molecules:
        try:
            dfx = fracDiff_FFD(data.to_frame(),d,thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx['price'], maxlag=1,regression='c',autolag=None)
            out.loc[d]=list(dfx[:4])+[dfx[4]['5%']]
        except Exception as e:
            print(f'{d} error: {e}')
    return out


def optimal_ffd(data, start = 0, end = 1, interval = 10, t=1e-5):
    
    for d in np.linspace(start, end, interval):    
        dfx = fracDiff_FFD(data.to_frame(), d, thres = t)
        if sm.tsa.stattools.adfuller(dfx['price'], maxlag=1,regression='c',autolag=None)[1] < 0.05:
            return d
    print('no optimal d')
    return d