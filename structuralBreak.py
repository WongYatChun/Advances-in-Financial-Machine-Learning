import pandas as pd 
import numpy as np

def lagDF(df0, lags):
    """
    Apply lags to data frame
    : params df0: pd.df, supposedly a pd series of log-prices differences (aka. returns)
    : params lags: the number of lags used in ADF specification, could be an integer or a list of integer
    
    : return df1: df that applied lags
    """
    # create an empty data frame
    df1 = pd.DataFrame()
    
    if isinstance(lags, int):
        # if `lags` is an integer
        # create a list of int from 0 to `lags`
        lags = range(lags+1)
    else:
        # if a list of lags has been specified
        # ensure they are ints and form the list
        lags = [int(lags) for lag in lags]
    
    for lag in lags: # for each lag
        # deep copy the 'lagged' df (log-prices) to df_ 
        df_ = df0.shift(lag).copy(deep=True)
        # column names "i_lags"
        df_.columns = [str(i) + '_' + str(lag) for i in df_.columns]
        # outer-join  df1 (combine the latest to df1)
        df1 = df1.join(df_, how = 'outer')
    return df1

def getYX(series, constant, lags):
    """
    Preparing the datasets
    : params series: pd.df, supposedly a pd series of log-prices
    : params constant: the regression's time trend component
                -   'nc': no time trend, only a constant
                -   'ct': a constant + a linear time trend
                -   'ctt': a constant + a 2nd degree polynomial time trend
    : params lags: the number of lags used in ADF specification, could be an integer or a list of integer
    
    : return y, x: 
    """
    # cal the log-prices differences (aka. returns), exclude the 1st value which is an NaN
    series_ = series.diff().dropna()
    # find all the lagged log-price returns
    x = lagDF(series_,lags).dropna()
    # normally the first column of x is the log-prices differences from t-1 to t if `lags` is a single int
    # first column x now becomes the lagged (last) log-prices
    x.iloc[:,0] = series.values[-x.shape[0]-1: -1, 0] # lagged level
    # y is the log-prices differences from t-1 to t
    y = series_.iloc[-x.shape[0]:].values

    if constant != 'nc':
        x = np.append(x,np.ones((x.shape[0],1)),axis=1)
        if constant[:2] == 'ct':
            # + a linear time trend
            trend = np.arange(x.shape[0]).reshape(-1,1)
            x = np.append(x, trend, axis=1)
        if constant == 'ctt':
            # + a 2nd degree polynomial time trend
            x = np.append(x, trend**2, axis = 1)
    return y, x

def getBetas(y,x):
    """
    carries out the actual regression
    : params y: pd.df, regressand, supposedly y is the log-prices differences from t-1 to t
    : params x: pd.df, regressor, supposedly x contains the last price and the lagged log-prices

    : return 
        bMean: pd.df, basically betas
        bVar: pd.df, betas' variances
    """
    xy = np.dot(x.T,y)
    xx = np.dot(x.T,x)
    xxinv = np.linalg.inv(xx)
    # betas
    bMean = np.dot(xxinv, xy)
    # prediction rrors
    err = y - np.dot(x,bMean)
    bVar = np.dot(err.T,err)/(x.shape[0] - x.shape[1]) * xxinv

    return bMean, bVar

def get_bsadf(logP, minSL, constant, lags):
    """
    estimate SADF = sup{beta/betaStd} which is the backshifting component of  the algorithm
    : params logP: pd.df, supposedly a pd series of log-prices
    : params minSL: float, the minimum sample length used by the final regression
    : params constant: the regression's time trend component
                -   'nc': no time trend, only a constant
                -   'ct': a constant + a linear time trend
                -   'ctt': a constant + a 2nd degree polynomial time trend
    : params lags: the number of lags used in ADF specification, could be an integer or a list of integer
    
    : return out: {'Time':logP.index[-1], 'gsadf':backshifting ADF}
    """
    # 1) Prepare the datasets
    y, x = getYX(logP, constant = constant, lags=lags)
    # the starting point starts from 0 (the beginning of the y data) to {no. of data + no. of lags - (minimum sample length used by the final regression - lags)}
    startPoints = range(0, y.shape[0] + lags - minSL + 1)
    # the variable store the maximum ADF
    bsadf = None
    # the list that store all the ADF
    allADF = []

    for start in startPoints: # for each start point
        # select a portion of the datasets from the start point to today
        y_, x_ = y[start:], x[start:]
        # 2) find the betas and the betas'volatilities
        bMean_, bStd_ = getBetas(y_, x_)
        # we only care about the 1st beta (which is the coefficient of the last price) and its std
        bMean_, bStd_ = bMean_[0,0], bStd_[0,0]**.5
        # calculate the ADF stat and store it into the allADF list
        allADF.append(bMean_/bStd_)
        # Dynamically update the maximum
        if allADF[-1] > bsadf:
            bsadf = allADF[-1]
    # output
    out = {'Time':logP.index[-1], 'gsadf':bsadf}
    return out