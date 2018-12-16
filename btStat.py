import pandas as pd 
import numpy as np

def getBets(tPos):
    """
    Derive the timing of bets from a series of target positions
    A bet takes place between flat positions or position flips
    A sequence of positions on the same side is considered part of the same bet
    A bet ends when the position is flattened or flipped to the opposite side
    """
    #----------------Find when position turns Flat----------------------
    # if the position == 0, the position is exited
    df0 = tPos[tPos == 0].index
    df1 = tPos.shift(1)
    # df0 is the date when position is not 0
    df1 = df1[df1 != 0].index
    # bets is the date when the last date's position is not 0 but the current position is 0
    bets = df0.intersection(df1) # flattening
    #----------------Find when position flips----------------------------
    # multiply the current bet position to the next bet position
    # if the result is negative, it means the position has been flipped
    df0 = tPos.iloc[1 : ] * tPos.iloc[ : -1].values
    # add those index with the negative values to the the set `bets`
    # dates are also sorted
    bets = bets.union(df0[df0 < 0].index).sort_values() # tPos flips
    
    if tPos.index[-1] not in bets: 
        # if the last index is not in the bet
        # 'tPos.index[-1 : ]': reserve the index datatype
        bets = bets.append(tPos.index[-1 : ]) # append the last date
    return bets

def getHoldingPeriod(tPos):
    """
    Derive avg holding period (in days) using avg entry time pairing algo
    Average holding period is the average no. of days(maybe miniutes) a bet is held
    """
    # create an empty df storing 'dT' and 'w'
    hp = pd.DataFrame(columns = ['dT', 'w']) 
    tEntry = 0.0 
    pDiff = tPos.diff() # position difference
    # time difference of each data point to the first avaialable date
    tDiff = (tPos.index - tPos.index[0]) / np.timedelta64(1,'s')

    for i in range(1, tPos.shape[0]): # for index 1 to the index of the last tPos
        if pDiff.iloc[i] * tPos.iloc[i - 1] >= 0: # if the position is increased or unchanged
            if tPos.iloc[i] != 0: # if the position ahead is not 0
                tEntry = (tEntry * tPos.iloc[i - 1] + tDiff[i] * pDiff.iloc[i]) / tPos.iloc[i]
        else: # if the position is decreased
            if tPos.iloc[i] * tPos.iloc[i - 1] < 0: # if the position has been flipped
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(tPos.iloc[i - 1]))
                tEntry = tDiff[i] # reset entry time
            else: # if the position has not been flipped
                hp.loc[tPos.index[i], ['dT', 'w']] = (tDiff[i] - tEntry, abs(pDiff.iloc[i]))
    if hp['w'].sum() > 0:
        hp = (hp['dT'] * hp['w']).sum() / hp['w'].sum() # average holding period
    else: # if no holding period
        hp = np.nan
    return hp

#————————————————————————————————————————
# ALGORITHM FOR DERIVING HHI CONCENTRATION
# rHHIPos = getHHI(ret[ret >= 0]) # concentration of positive returns per bet
# rHHINeg = getHHI(ret[ret < 0]) # concentration of negative returns per bet
# tHHI = getHHI(ret.groupby(pd.TimeGrouper(freq = 'M')).count()) # concentr. bets/month

def getHHI(betRet):
    # get the concentration of the (signed or unsigned) returns per bet
    if betRet.shape[0] <= 2: # if less than 3 bets
        return np.nan # no HHI
    wght = betRet / betRet.sum() # weight per return
    hhi = (wght**2).sum() 
    hhi = (hhi - betRet.shape[0]**-1)/(1. - betRet.shape[0]**-1)
    return hhi

def computeDD_TuW(series, dollars = False):
    """
    compute series of drawdowns and the time under water associated with them
    Drawdown: the max. loss suffered by an investment between two consecutive high-watermarks
    Time under Water (TuW): the time elapsed between an HWM and the moment the PnL exceeds the previous max PnL
    """
    # the series becomes a dataframe and the column is named as 'pnl'
    df0 = series.to_frame('pnl') 
    # it marks the data by the previous max.data, i.e. if there is a new highest price, all the prices after are under its hwm until seeing the next highest prices
    df0['hwm'] = series.expanding().max()
    # find the min. price/return of each hwm, reset the index (because groupby('hwm') will make the index become the 'hwm')
    df1 = df0.groupby('hwm').min().reset_index()
    # rename the columns
    df1.columns = ['hwm', 'min']
    # df1.index is the first date of the hwm
    df1.index = df0['hwm'].drop_duplicates(keep = 'first').index # time of hwm
    # only find those hwm followed by a drawdown
    df1 = df1[df1['hwm'] > df1['min']] 
    if dollars: ## the series of dollar performance
        dd = df1['hwm'] - df1['min']
    else: # the series of returns
        dd = 1 - df1['min'] / df1['hwm']
    # find the duration of the hwm (Time under Water)
    tuw = ((df1.index[1 : ] - df1.index[ : -1]) / np.timedelta64(1, 's')).values# in seconds
    # hold in a pd.series
    tuw = pd.Series(tuw, index = df1.index[ : -1])
    return dd, tuw