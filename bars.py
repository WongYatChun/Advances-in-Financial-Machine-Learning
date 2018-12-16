import pandas as pd 
import numpy as np 
import scipy.stats as stats
from numba import jit 
from mpEngine import mpPandasObj
from utils import returns, read_bbg_ticks, getDailyVol
from tqdm import tqdm, tqdm_notebook


def general_bars(df,column,m,tick = False):
    """
    Compute tick bars
    
    # Args:
        df: pd.DataFrame()
        column: name for price data
        m: int(), threshold value for ticks
    # Returns:
        idx: list of indices
    """
    t = df[column]
    ts = 0
    idx = []
    # for i, x in enumerate(tqdm(t)):
    if tick: # if tick bar
        for i,x in enumerate(t):
            ts += 1 # each data plus 1
            if ts > m:
                idx.append(i)
                ts = 0
                # continue
    else: # if not tick bar
        for i,x in enumerate(t):
            ts += x # each data plus volume/dollar volume
            if ts > m:
                idx.append(i)
                ts = 0
                # continue
    return idx

def general_bar_df(df,column,m, tick = False):
    idx = general_bars(df, column, m, tick)
    df = df.iloc[idx].drop_duplicates()
    df['dates'] = df['dates'] + pd.to_timedelta(df.groupby('dates').cumcount(), unit='ms')
    return df

def tick_bars(df,price_column,m):
    return general_bars(df,price_column,m, tick = True)

def volume_bars(df,volume_column,m):
    return general_bars(df,volume_column,m)

def dollar_bars(df,dollar_column,m):
    return general_bars(df,dollar_column,m)


def tick_bar_df(df,tick_column,m):
    return general_bar_df(df,tick_column,m,tick = True)

def volume_bar_df(df,volume_column,m):
    return general_bar_df(df,volume_column,m)

def dollar_bar_df(df,dollar_column,m):
    return general_bar_df(df,dollar_column,m)

def get_ohlc(ref,sub):
    """
    fn: get ohlc from custom
    
    # args
        ref: reference pandas series with all prices
        sub: custom tick pandas series
    
    # returns
        tick_df: dataframe with ohlc values
    """
    ohlc = []
    # for i in tqdm(range(sub.index.shape[0]-1)):
    for i in range(sub.index.shape[0] - 1):
        start, end = sub.index[i], sub.index[i+1]
        tmp_ref = ref.loc[start:end]
        max_px, min_px = tmp_ref.max(), tmp_ref.min()
        o, h, l, c = sub.iloc[i], max_px, min_px, sub.iloc[i+1]
        ohlc.append((end, start, o, h, l,c))
    cols = ['end','start','open','high','low','close']
    return (pd.DataFrame(ohlc,columns=cols))

@jit(nopython = True)
def numba_isclose(a,b,rel_tol = 1e-09, abs_tol=0.0):
    # rel_tol: relative tolerance
    # abs_tol: absolute tolerance
    return np.fabs(a-b) <= np.fmax(rel_tol*np.fmax(np.fabs(a),np.fabs(b)),abs_tol)

@jit(nopython=True)
def bt(p0,p1,bs):
    """
    Determine the direction of the tick
    using tick rule
    """
    if numba_isclose((p1-p0),0.0,abs_tol=0.001):
        return bs[-1]
    else: return np.abs(p1-p0)/(p1-p0)

def get_imbalance(t):
    """Noted that this function return a list start from the 2nd obs"""
    bs = np.zeros_like(t)
    for i in np.arange(1,bs.shape[0]):
        bs[i-1] = bt(t[i-1],t[i],bs[:i-1])
    return bs[:-1] # remove the last value

def test_t_abs(absTheta,t,E_bs):
    """
    Bool function to test inequality
    * row is assumed to come from df.itertuples()
    - absTheta: float(), row.absTheta
    - t: pd.Timestamp
    - E_bs: float, row.E_bs
    """
    return (absTheta >= t*E_bs)

def agg_imalance_bars(df):
    """
    Implements the accumulation logic
    """
    start = df.index[0]
    bars = []
    for row in df.itertuples():
        t_abs = row.absTheta
        rowIdx = row.Index
        E_bs = row.E_bs
        
        t = df.loc[start:rowIdx].shape[0]
        if t<1: t = 1 # if t less than 1, set equal to 1
        if test_t_abs(t_abs,t,E_bs):
            bars.append((start,rowIdx,t))
            start = rowIdx
    return bars

def retClose(df):
    df['retClose'] = returns(df)
    return df

def orderFlow(df, degreeFree = 9999999,window = 100):
    """
    :param df: pd.df, assume has ['retClose','dailyVol'] = ['return of the close price','volatility']
    :param degreeFree: constant, degree of freedom for t-distribution
    :param window: rolling window for VPIN
    :return: df ['normOI','VPIN']
    """
    # normalised Order Imbalance
    df['normOI'] = 2*stats.t.cdf(df['retClose']/df['dailyVol'],degreeFree)-1
    # calculate VPIN
    df['VPIN'] = df['normOI'].rolling(span = window).mean()
    return df

def vpinPre(fp, m=False,volSpan = 100,degreeFree = 9999999, vpinSpan = 100):
    """
    :param fp: file path of the raw data
    :param m: volume sample size
    :param volSpan: daily vol span
    :param degreeFree: degree of freedom for t-distribution, default: infinity -> normal distribution
    :param vpinSpan: VPIN span
    :return: df['dates','volume','price','retClose','dailyVol','normOI','VPIN']
    """
    # read bbg data
    df = read_bbg_ticks(fp)
    # calc volume sample size
    if m == False:
        # if not specified, average volume is used
        m = df.volume.sum()/df.shape[0]
    # cal volume bar

    df = volume_bar_df(df,'volume', m)
    df.set_index('dates',inplace = True)
    # create return bar
    df['retClose'] = df['price']/df['price'].shift(1) - 1

    # cal daily volatility of return
    df['dailyVol'] = getDailyVol(df['price'],span0=volSpan)
    # cal normalised Order Imbalance#
    df['normOI'] = 2 * stats.t.cdf(df['retClose'] / df['dailyVol'], degreeFree) - 1
    # cal VPIN
    df['VPIN'] = df['normOI'].rolling(vpinSpan).mean()

    return df

