import pandas as pd 
import numpy as np 
import scipy.stats as stats
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit 



plt.style.use('seaborn-talk')
plt.style.use('bmh')
plt.rcParams['font.weight'] = 'medium'

def cprint(df):
    if not isinstance(df, pd.DataFrame):
        try:
            df = df.to_frame()
        except:
            raise ValueError('object cannot be coerced to df')

    print('-'*79)
    print('dataframe information')
    print('-'*79)
    print(df.tail(5))
    print('-'*50)
    print(df.info())
    print('-'*79)
    print()

get_range = lambda df, col: (df[col].min(), df[col].max())

def read_bbg_ticks(fp):
    raw_df = pd.read_csv(fp)
    df = (raw_df
            .assign(dates = lambda raw_df: pd.to_datetime(raw_df['Unnamed: 0']))
            .assign(volume = lambda raw_df: raw_df['size'])
            .assign(price = lambda raw_df: raw_df['value'])
            .drop(['Unnamed: 0','type','size','value'],axis = 1)
            #.set_index('dates')
         )
    return df


def read_kibot_ticks(fp):
    # read tick data
    cols = list(map(str.lower,['Date','Time','Price','Bid','Ask','Size']))
    df = (pd.read_csv(fp, header = None)
          .rename(columns = dict(zip(range(len(cols)),cols)))
          .assign(dates = lambda df : (pd.to_datetime(df['date']+df['time'],
                                                      format = '%m/%d/%Y%H:%M:%S')))
          .assign(v = lambda df: df['size']) # volume
          .assign(dv = lambda df: df['price'] * df['size']) # dollar volume
          .drop(['date','time'],axis = 1)
          .set_index('dates')
          .drop_duplicates())
    return df


@jit(nopython=True)
def mad_outlier(y, thresh=3.):
    '''
    compute outliers based on mad
    # args
        y: assumed to be array with shape (N,1)
        thresh: float()
    # returns
        array index of outliers
    '''
    # This function is an approximation of MAD Approach
    # @jit can speed up the calculation but may not be applicable to the orginal MAD Approach
    median = np.median(y)
    print(median)
    diff = np.sum((y - median)**2, axis=-1) # this line is complicated
    diff = np.sqrt(diff)
    print(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation
    print(modified_z_score)
    
    return modified_z_score > thresh

def getDailyVol (close, span0 = 100):
    """
    Compute the daily volatility at intraday estimation
    applying a span of span0 to an exponentially weighted moving standard deviation
    
    Set profit taking and stop loss limits that are function of the risks involved in a bet
    """

    df0 = close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0 = df0[df0>0]   
    df0 = (pd.Series(close.index[df0-1], 
                   index=close.index[close.shape[0]-df0.shape[0]:]))   
    try:
        df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0=df0.ewm(span=span0).std().rename('dailyVol')
    return df0



def select_sample_data(ref,sub,price_col,date):
    """
    select a sample of data based on data, assumes datatimeindex
    
    # args
        ref: pd.DataFrame containing all ticks
        sub: subordinated pd.DataFrame of prices
        price_col: str(), price colume
        date: str(), date to select
    
    # returns
        xdf: ref pd.Series
        xtdf: subordinated pd.Series
    """
    
    xdf = ref[price_col].loc[date]
    xtdf = sub[price_col].loc[date]
    
    return xdf, xtdf

def plot_sample_data(ref, sub, bar_type, *args, **kwds):
    _, axes = plt.subplots(3,sharex=True, sharey=True, figsize=(10,7))
    
    ref.plot(*args, **kwds, ax=axes[0], label='price')
    sub.plot(*args, **kwds, ax=axes[0], marker='X', ls='', label=bar_type)
    axes[0].legend()
    
    ref.plot(*args, **kwds, ax=axes[1], marker='o', label='price')
    sub.plot(*args, **kwds, ax=axes[2], marker='X', ls='', 
             color = 'r', label=bar_type)
    
    for ax in axes[1:]:ax.legend()
    
    plt.tight_layout()
    
    return

def scale(s):
    """Standardize the data for comparison"""
    return (s - s.min())/(s.max()-s.min())

def returns(s):
    """Compute the log return of the s"""
    arr = np.diff(np.log(s))
    return (pd.Series(arr, index = s.index[1:]))

def get_test_stats(bar_types, bar_returns, test_func, 
                   *args, **kwds):
    dct = {bar:(int(bar_ret.shape[0]),test_func(bar_ret, *args, **kwds))
           for bar, bar_ret in zip(bar_types, bar_returns)}
    df = (pd.DataFrame.from_dict(dct)
          .rename(index={0:'sample size',1:f'{test_func.__name__}_stat'}).T)
    
    return df

def plot_autocorr(bar_types, bar_returns):
    f, axes = plt.subplots(len(bar_types),figsize=(10,7))
    
    for i, (bar,typ) in enumerate(zip(bar_returns, bar_types)):
        sm.graphics.tsa.plot_acf(bar, lags=120, ax = axes[i],
                                 alpha = 0.05, unbiased = True, fft = True,
                                 zero = False,
                                 title = f'{typ} AutoCorr')
    plt.tight_layout()

def plot_hist(bar_types, bar_ret):
    f, axes = plt.subplots(len(bar_types), figsize=(10,6))
    for i, (bar, typ) in enumerate(zip(bar_ret, bar_types)):
        g = sns.distplot(bar, ax=axes[i], kde = False, label = typ)
        g.set(yscale='log')
        axes[i].legend()
    plt.tight_layout()

def jb(x, test=True):
    #np.random.seed(12345678)
    if test: return stats.jarque_bera(x)[0]
    return stats.jarque_bera(x)[1]

def shapiro(x, test = True):
    if test: return stats.shapiro(x)[0]
    return stats.shapiro(x)[1]