import pandas as pd 
import numpy as np
from mpEngine import mpPandasObj, processJobs, processJobs_


def mpNumCoEvents(closeIdx, t1, molecule):
    """
    Compute the number of concurrent events per bar
    :params closeIdx: pd.df, the index of the close price
    :param t1: pd series, timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
    :param molecule: the date of the event on which the weight will be computed
        + molecule[0] is the date of the first event on which the weight will be computed
        + molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[molecule].max() impacts the count
    :return:
        count: pd.Series, the number of concurrent event per bar
    """
    # 1) Find events that span the period [molecule[0], molecule[-1]]
    # unclosed events still impact other weights
    # fill the unclosed events with the last available (index) date
    t1 = t1.fillna(closeIdx[-1]) 
    # events that end at or after molecule[0] (the first event date)
    t1 = t1[t1 >= molecule[0]]
    # events that start at or before t1[molecule].max() which is the furthest stop date in the batch
    t1 = t1.loc[ : t1[molecule].max()]

    # 2) Count events spanning a bar
    # find the indices begining start date ([t1.index[0]) and the furthest stop date (t1.max())
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    # form a 0-array, index: from the begining start date to the furthest stop date
    count = pd.Series(0, index = closeIdx[iloc[0] : iloc[1] + 1])
    # for each signal t1 (index: eventStart, value: eventEnd)
    for tIn, tOut in t1.iteritems():
        # add 1 if and only if [t_(i,0), t_(i.1)] overlaps with [t-1,t]
        count.loc[tIn : tOut] += 1 # every timestamp between tIn and tOut
    # compute the number of labels concurrents at t
    return count.loc[molecule[0] : t1[molecule].max()] # only return the timespan of the molecule

def mpSampleTW(t1, numCoEvents, molecule):
    """
    :param t1: pd series, timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
    :param numCoEvent: 
    :param molecule: the date of the event on which the weight will be computed
        + molecule[0] is the date of the first event on which the weight will be computed
        + molecule[-1] is the date of the last event on which the weight will be computed
    :return
        wght: pd.Series, the sample weight of each (volume) bar
    """
    # derive average uniqueness over the event's lifespan
    wght = pd.Series(index = molecule)
    # for each events
    for tIn, tOut in t1.loc[wght.index].iteritems():
        # tIn, starts of the events, tOut, ends of the events
        # the more the coEvents, the lower the weights
        wght.loc[tIn] = (1. / numCoEvents.loc[tIn : tOut]).mean()
    return wght

def SampleTW(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    :return
        wght: pd.Series, the sample weight of each (volume) bar
    """
    out = events[['t1']].copy(deep = True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(mpNumCoEvents, ('molecule', events.index), numThreads, closeIdx = close.index, t1 = out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep = 'last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(mpSampleTW, ('molecule', events.index), numThreads, t1 = out['t1'], numCoEvents = numCoEvents)
    return out

def getIndMatrix(barIx, t1):
    """
    Get Indicator matrix
    :param barIx: the index of bars
    :param t1: pd series, timestamps of the vertical barriers. (index: eventStart, value: eventEnd).
    :return indM: binary matrix, indicate what (price) bars influence the label for each observation
    """
    indM = pd.DataFrame(0, index = barIx, columns = range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()): # signal = obs
        indM.loc[t0 : t1, i] = 1. # each obs each column, you can see how many bars are related to an obs/
    return indM

def getAvgUniqueness(indM):
    """
    Get Indicator matrix
    :param indM: binary matrix, indicate what (price) bars influence the label for each observation
    :return avgU: average uniqueness of each observed feature
    """
    # Average uniqueness from indicator matrix
    c = indM.sum(axis = 1) # concurrency, how many obs share the same bar
    u = indM.div(c, axis = 0) # uniqueness, the more obs share the same bar, the less important the bar is
    avgU = u[u > 0].mean() # average uniquenessn
    return avgU

def seqBootstrap(indM, sLength = None):
    """
    Give the index of the features sampled by the sequential bootstrap
    :param indM: binary matrix, indicate what (price) bars influence the label for each observation
    :param sLength: optional, sample length, default: as many draws as rows in indM
    """
    # Generate a sample via sequential bootstrap
    if sLength is None: # default
        sLength = indM.shape[1] # sample length = # of rows in indM
    # Create an empty list to store the sequence of the draws
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series() # store the average uniqueness of the draw
        for i in indM: # for every obs
            indM_ = indM[phi + [i]] # add the obs to the existing bootstrapped sample
            # get the average uniqueness of the draw after adding to the new phi
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1] # only the last is the obs concerned, others are not important
        prob = avgU / avgU.sum() # cal prob <- normalise the average uniqueness
        phi += [np.random.choice(indM.columns, p = prob)] # add a random sample from indM.columns with prob. = prob
    return phi

def main():
    # t0: t1.index; t1: t1.values
    t1 = pd.Series([2, 3, 5], index = [0,2,4])
    # index of bars
    barIx = range(t1.max() + 1)
    # get indicator matrix
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size = indM.shape[1])
    print (phi)
    print ('Standard uniqueness:', getAvgUniqueness(indM[phi]).mean())
    phi = seqBootstrap(indM)
    print (phi)
    print ('Sequential uniqueness:', getAvgUniqueness(indM[phi]).mean())

if __name__ == "__main__": main()

def getRndT1(numObs, numBars, maxH):
    # random t1 Series
    t1 = pd.Series()
    for _ in range(numObs):
        ix = np.random.randint(0, numBars)
        val = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
    return t1.sort_index()

def auxMC(numObs, numBars, maxH):
    # Parallelized auxiliary function
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1.max() + 1)
    indM = getIndMatrix(barIx, t1)
    phi = np.random.choice(indM.columns, size = indM.shape[1])
    stdU = getAvgUniqueness(indM[phi]).mean()
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return {'stdU': stdU, 'seqU': seqU}

def mainMC(numObs = 10, numBars = 100, maxH = 5, numIters = 1E6, numThreads = 24):
    # Monte Carlo experiments
    jobs=[]
    for _ in range(int(numIters)):
        job={'func': auxMC, 'numObs': numObs, 'numBars': numBars, 'maxH': maxH}
        jobs.append(job)
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads = numThreads)
    print (pd.DataFrame(out).describe())
    return

def mpSampleW(t1, numCoEvents, close, molecule):
    # Derive sample weight by return attribution
    ret = np.log(close).diff() # log-returns, so that they are additive
    wght = pd.Series(index = molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn : tOut] / numCoEvents.loc[tIn : tOut]).sum()
    return wght.abs()


def SampleW(close, events, numThreads):
    """
    :param close: A pd series of prices
    :param events: A Pd dataframe
        -   t1: the timestamp of vertical barrier. if the value is np.nan, no vertical barrier
        -   trgr: the unit width of the horizontal barriers, e.g. standard deviation
    :param numThreads: constant, The no. of threads concurrently used by the function
    """
    out = events[['t1']].copy(deep = True)
    numCoEvents = mpPandasObj(mpNumCoEvents,('molecule', events.index),numThreads, closeIdx = close.index, t1 = events['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep = 'last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1 = events['t1'], numCoEvents = numCoEvents, close = close)
    out['w'] *= out.shape[0] / out['w'].sum() # normalised, sum up to sample size
    
    return out

def get_Concur_Uniqueness(close, events, numThreads):
    out = events[['t1']].copy(deep = True)
    out['t1'] = out['t1'].fillna(close.index[-1])
    events['t1'] = events['t1'].fillna(close.index[-1])
    numCoEvents = mpPandasObj(mpNumCoEvents, ('molecule', events.index), numThreads, closeIdx = close.index, t1 = out['t1'])
    numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep = 'last')]
    numCoEvents = numCoEvents.reindex(close.index).fillna(0)
    out['tW'] = mpPandasObj(mpSampleTW, ('molecule', events.index), numThreads, t1 = out['t1'], numCoEvents = numCoEvents)
    out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1 = events['t1'], numCoEvents = numCoEvents, close = close)
    out['w'] *= out.shape[0] / out['w'].sum() # normalised, sum up to sample size
    return out
    
def getTimeDecay(tW, clfLastW = 1.):
    """
    apply piecewise-linear decay to observed uniqueness (tW)
    clfLastW = 1: no time decay
    0 <= clfLastW <= 1: weights decay linearly over time, but every obersevation still receives a strictly positive weight
    c = 0: weughts converge linearly to 0 as they become older
    c < 0: the oldest portion cT of the observations receive 0 weight
    c > 1: weights increase as they get older"""
    # newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW = tW.sort_index().cumsum() # cumulative sum of the observed uniqueness
    if clfLastW >= 0: # if 0 <= clfLastW <= 1
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else: # if -1 <= clfLastW < 1
        slope=1. / ((clfLastW + 1) * clfW.iloc[-1])
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0 # neg weight -> 0
    print (const,slope)
    return clfW