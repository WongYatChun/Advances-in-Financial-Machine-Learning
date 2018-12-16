import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from numba import jit
from mpEngine import mpPandasObj, mpJobList


def tradableHour(i, start = '09:40', end = '15:50'):
    """
    : param i: a datetimeIndex value
    : param start: the start of the trading hour
    : param end: the end of the trading hour
    
    : return: bool, is tradable hour or not"""
    time = i.strftime('%H:%M')
    return (time < end and time > start)

def getTEvents(gRaw, h, symmetric = True, isReturn = False):
    """
    Symmetric CUSUM Filter
    Sample a bar t iff S_t >= h at which point S_t is reset
    Multiple events are not triggered by gRaw hovering around a threshold level
    It will require a full run of length h for gRaw to trigger an event
    
    Two arguments:
        gRaw: the raw time series we wish to filter (gRaw), e.g. return
        h: threshold
        
    Return:
        pd.DatatimeIndex.append(tEvents): 
    """
    tEvents = []
    if isReturn:
        diff = gRaw
    else:
        diff = gRaw.diff()
    if symmetric:
        sPos, sNeg = 0, 0
        if np.shape(h) == ():

            for i in diff.index[1:]:
                sPos, sNeg = max(0,sPos+diff.loc[i]), min(0,sNeg+diff.loc[i])
                if sNeg < -h and tradableHour(i):
                    sNeg = 0; tEvents.append(i)
                elif sPos > h and tradableHour(i):
                    sPos = 0; tEvents.append(i)
        else:
            for i in diff.index[1:]:
                sPos, sNeg = max(0,sPos+diff.loc[i]), min(0,sNeg+diff.loc[i])
                if sNeg < -h[i] and tradableHour(i):
                    sNeg = 0; tEvents.append(i)
                elif sPos > h[i] and tradableHour(i):
                    sPos = 0; tEvents.append(i)
    else:
        sAbs = 0
        if np.shape(h) == ():
            
            for i in diff.index[1:]:
                sAbs = sAbs + diff.loc[i]
                if sAbs > h and tradableHour(i):
                    sAbs = 0; tEvents.append(i)
                
        else:
            for i in diff.index[1:]:
                sAbs = sAbs+diff.loc[i]
                if sAbs > h[i] and tradableHour(i):
                    sAbs = 0; tEvents.append(i)
            
    return pd.DatetimeIndex(tEvents)

def applyPtSlOnT1(close, events, ptSl, molecule):
    """
    apply stop loss/profit taking, if it takes place before t1 (end of event)
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param ptSl: A list of two non-negative float values
        -   ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier (if 0, no uBarrier)
        -   ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier (if 0, no lBarrier)
    :param moledule: A list with the subset of event indices that will be processed by a single thread
        -   for multi-thread
    :return: A pandas dataframe containing the timestamps at which each barrier was touched
    """
    events_ = events.loc[molecule] # multiprocessing
    # a copy of the pd series of the start times (index) and the stop times (values) of the events
    out = events_[['t1']].copy(deep = True)
    if ptSl[0] > 0:
        # an pd series stores the uBarrier
        pt = ptSl[0] * events_['trgt']
    else:
        # an NaN pd series that has the same index of the events
        pt = pd.Series(index = events.index) # NaNs,
    if ptSl[1] > 0:
        # an pd series stores the lBarrier
        sl = - ptSl[1] * events_['trgt']
    else:
        # an NaN pd series that has the same index of the events
        sl = pd.Series(index = events.index)  # NaNs,

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        # fillna(close.index[-1]): if vBarrier is not specified, then it is the last date of the dataset

        # all the prices from the start of the events to the end of the events
        df0 = close[loc:t1] # df0: path prices
        # calculate the returns incurred by each single price
        df0 = (df0 / close[loc]-1) * events_.at[loc, 'side'] # df0: path returns

        # Find all the returns exceed the horinzontal barriers (Stop loss/Profit taking)
        # Stores the earliest (minimum) timestamp(s) when limit(s) are exceeded
        out.loc[loc,'sl'] = df0[df0 < sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking
    return out


def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    """
    Find the time of the first barrier touch
    :param close: A pd series of prices
    :param tEvents: The pd timeindex containing the timestamps that produce triple barriers
    :param ptSl: A non-neg float that sets the width of the two barriers. 0 means the respective horizontal barrier will be disabled
    :param trgt: A pd series of targets, expressed in terms of absolute returns
    :param minRet: constant, The min target return required for the running a triple barrier search
    :param numThreads: constant, The no. of threads concurrently used by the function
    :param t1: A pd series with the timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
        -   If trgt = False, vBarrier is disabled
    :param side: A pd series with the timestamps of the vertical barriers. (eventStart, eventEnd). 
        -   If trgt = False, vBarrier is disabled

    :return: A pd dataframe with columns:
        -   Index: time when 1) trgt > threshold, 2) triple barriers are triggered
        -   t1: the timestamps at which the first barrier is touched
        -   trgt: the target that was used to generate the horizontal barriers
        -   side (optional): the side of the trade
    """
    #1) get target
    trgt=trgt.loc[tEvents] # get a list of targets when triple barriers are triggered
    trgt=trgt[trgt>minRet] # only select those trible barriers events when the targets are above a certain threshold
    
    #2) get t1 (max holding period)
    if t1 is False: # if no limit on holding period (no vBarriers)
        t1 = pd.Series(pd.NaT, index=tEvents) # create an NaT pd.series that havs the same index as the pd.series of tEvents

    #3) form events object, apply stop loss on t1
    if side is None: # if side is not fed into the function
        side_ = pd.Series(1.,index=trgt.index) # create a pd.series of '1' that havs the same index as the pd.series of trgt
        ptSl_ = [ptSl[0],ptSl[0]] # assume symmetric barriers, uBarrier and lBarrier have the same width
    else:
        side_ =side.loc[trgt.index] # only select those sides when 1) trgt > threshold, 2) triple barriers are triggered
        ptSl_ = ptSl[:2] # barriers are the same as the input

    # I think the index of `events` is the same as the trgt
    # events: 'index',' t1', 'trgt', 'side'
    events=(pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1)
            .dropna(subset=['trgt']))
    
    # Multiprocessing
    # Input: 1) close prices, 2) events' timestamp, 3) the width of the barriers
    # Output: df0, A pandas dataframe containing the timestamps at which each barrier was touched
    # df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule',events.index),
    #                 numThreads=numThreads, close=close, events=events,
    #                 ptSl=ptSl_)

    df0 = mpJobList(func=applyPtSlOnT1, argList =('molecule', events.index),
                      numThreads=numThreads, redux = pd.DataFrame.append, close=close, events=events,
                      ptSl=ptSl_)

    # Drop the data in df0 if no limit is touched
    # Find the earliest of the three dates (NaN is igorned)
    events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    
    # If `side` is not fed, drop the column
    if side is None:
        events=events.drop('side',axis=1)
    
    return events

def addVerticalBarrier(tEvents, close, hour = False):
    """
    Intraday version
    find the timestamp of the next price bar at or immediately after a number of days `numDays`
    :param tEvents: The pd timeindex containing the timestamps that will seed every triple barrier
    :param close: A pd series of prices
    :param numDays: the no. of days after the current
    :return: pd.series, the vertical barrier, {index: tEvents, value: tEvents + numDays}
    """
    if hour != False:
        # find the index of the timestamp of the next price bar at or immediately after a number of days `numDays`
        t1 = close.index.searchsorted(tEvents + pd.Timedelta(hours = hour))
    else:
        # very clumsy, need to find a better solution
        t1 = close.index.searchsorted(tEvents + pd.Timedelta(hours = 8))
        
    # Only return those index that are less or equal to the the max. of the `close`
    t1 = t1[t1 < close.shape[0]]
    # t1: pd.series. {index: tEvents, value: tEvents + numDays}
    t1 = (pd.Series(close.index[t1], index = tEvents[ : t1.shape[0]]))
    return t1

def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    # drop the events with no t1 (should have cleaned before enter)
    events_=events.dropna(subset=['t1'])
    # px is the list of timestamps that have either 'start' or 'end' events
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    # find all the prices when there is a either 'start' or 'end' event
    px=close.reindex(px,method='bfill')

    #2) create out object
    # create a pd.df with the indexes same as events_.index
    out=pd.DataFrame(index=events_.index)
    # create a `ret` column that calculate the returns from the 'end' event to the 'start' events
    # it calculates the returns of signals
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1

    if 'side' in events_:
        # if there is a column 'side' in the 'events'
        # flip the signs of the return corresponding to the side
        # i.e. if neg returns and short-selling = pos returns
        out['ret']*=events_['side'] # meta-labeling

    # create a 'bin' column, which is the sign of the returns
    out['bin']=np.sign(out['ret'])

    if 'side' in events_:
        # if there is a column 'side' in the 'events'
        # turn all the ['bin'] to 0 if ret <= 0
        # since the side is fed into the program, the output 'bin' only cares whether trade or pass
        out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out

def dropLabels(events, minPtc = .05):
    # apply weights, drop labels with insufficient examples
    while True:
        # count the labels and normalise the count
        df0=events['bin'].value_counts(normalize=True)
        # if the smallest value of the no. of labels is larger than the threshold
        #   then all labels are significant, break
        # elseif only 2 (or less) labels left,
        #   then break
        if df0.min()>minPtc or df0.shape[0]<3:break
        print('dropped label: ', df0.argmin(),df0.min())
        # drop the data with the insignificant label from the `events`
        events=events[events['bin']!=df0.argmin()]

    return events



def getBinsOld(events, close):
    """
    label the observations
    :param events: a pd dataframe of the time of the first barrier touch
        -   t1: the timestamps at which the first barrier is touched
        -   trgt: the target that was used to generate the horizontal barrier
    :param close: pd series of prices
    :return: a pd dataframe with columns
        -   ret: The return realised at the time of the first touched barrier
        -   bin: The label, {-1, 0, 1}, as a function of the sign of the outcome
    """
    #1) prices aligned with events
    events_=events.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin']=np.sign(out['ret'])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query('index in @t1').index
        out.loc[locs, 'bin'] = 0
    except:
        pass
    return out

# def getTEvents(gRaw, h):
    #   # if the input is a series of prices and h is standard deviation of returns (they are not measuring the same thing)
    # tEvents, sPos, sNeg = [], 0, 0
    # diff = np.log(gRaw).diff().dropna().abs()
    # for i in tqdm(diff.index[1:]):
    #     try:
    #         pos, neg = float(sPos+diff.loc[i]), float(sNeg+diff.loc[i])
    #     except Exception as e:
    #         print(e)
    #         print(sPos+diff.loc[i], type(sPos+diff.loc[i]))
    #         print(sNeg+diff.loc[i], type(sNeg+diff.loc[i]))
    #         break
    #     sPos, sNeg=max(0., pos), min(0., neg)
    #     if sNeg<-h:
    #         sNeg=0;tEvents.append(i)
    #     elif sPos>h:
    #         sPos=0;tEvents.append(i)
    # return pd.DatetimeIndex(tEvents)

    # def addVerticalBarrier(tEvents, close, numDays=1):
    #     """
    #     Interday version
    #     find the timestamp of the next price bar at or immediately after a number of days `numDays`
    #     :param tEvents: The pd timeindex containing the timestamps that will seed every triple barrier
    #     :param close: A pd series of prices
    #     :param numDays: the no. of days after the current
    #     :return: pd.series, the vertical barrier, {index: tEvents, value: tEvents + numDays}
    #     """
    #     # find the index of the timestamp of the next price bar at or immediately after a number of days `numDays`
    #     t1=close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
    #     # Only return those index that are less or equal to the the max. of the `close`
    #     t1=t1[t1<close.shape[0]]
    #     # t1: pd.series. {index: tEvents, value: tEvents + numDays}
    #     t1=(pd.Series(close.index[t1],index=tEvents[:t1.shape[0]]))
        
    #     return t1