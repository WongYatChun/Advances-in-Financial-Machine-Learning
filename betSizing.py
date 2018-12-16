import pandas as pd

from scipy.stats import norm
from mpEngine import mpPandasObj



# From probabilities to bet size
def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):
    """
    From probabilities to bet size
    : param events: pd.df with 
        Index: time (of the events, start of the signal) 
        t1: the time at which the first barrier is touched
        trgt: the target that was used to generate the horizontal barriers (but not used)
        side(optional): the side of the trade
    : param stepSize: constant (0,1], degree of discretization
    : param prob: constant, [0,1], the probability of the predicted outcome
    : param pred: constant, {-1...0...1}, the predicted outcomes
    : param numClasses: constant, the no. of possible outcomes/classes, default binary
    : param numThreads: constant, for multithread
    : param **kargs
    
    : return: constant, bet size
    """
    # get signals from predictions
    # if no prob, then return an empty pd.series
    if prob.shape[0] == 0:
        return pd.Series()
    
    # 1) Generate signals from multinomial classification (one-vs-rest, OvR)
    # t-value of OvR (test statistics), z
    signal0 = (prob - 1. / numClasses)/ (prob * (1.-prob))**.5
    # bet size (signal) = side * size, (CDF(is) - CDF(isnot)) * side
    signal0 = pred*(2*norm.cdf(signal0)-1)
    
    # meta lebeling
    # if side is included in the input 'events'
    # flip the direction of the bet size if necessary
    if 'side' in events:
        signal0 *= events.loc[signal0.index, 'side'] 

    # convert signal0 from series to dataframe df0 and the col named 'signal'
    # join the t1 to the df0, the index is the same as the df0
    df0 = signal0.to_frame('signal').join(events[['t1']],how = 'left')
    
    # 2) Compute average singal among those concurrently open
    df0 = avgActiveSignals(df0, numThreads)
    
    # 3) Discretize the final values
    signal1 = discreteSignal(signal0 = df0, stepSize = stepSize)

    return signal1

def avgActiveSignals(signals, numThreads):
    """
    compute the average signal among those active
    : param signals: a pd.df with
                Index: time, the start of the signal
                'signal': the betsize
                't1': the time at which the first barrier is touched 
    : param numThreads: for multithreading

    : return: pd.df, the average signal among those still active trade, or 0 if not active
    """
    # 1) time points where signals change (either one starts or one ends)
    # collect all the unique time points from signal['t1']
    tPnts = set(signals['t1'].dropna().values)
    # append all the unique time points (not exist in 't1') from signal.index 
    tPnts = tPnts.union(signals.index.values)
    # convert the set into a list
    tPnts = list(tPnts)
    # sort the list by chronological order
    tPnts.sort()
    # involke mpAvgActiveSignals and multiprocessing (or multithreading)
    out = mpPandasObj(mpAvgActiveSignals,
                        ('molecule',tPnts),
                        numThreads,
                        signals = signals)
    return out

def mpAvgActiveSignals(signals, molecule):
    """
    At time loc, average signal among those still active
    Signal is active if (there is a trade(s))
        a) issued before or at loc
        b) loc before signal's endtime, or endtime is still unknown (NaT)
    : param signals: a pd.df with
                Index: time, the start of the signal
                'signal': the betsize
                't1': the time at which the first barrier is touched 
    : param molecule: for multithreading

    : return: pd.df, the average signal among those still active trade, or 0 if not active
    """    
    # create an empty series, in case no loc
    out = pd.Series()
    # for each events (in the molecule)
    for loc in molecule:
        # determine whether the trade is active (by the above criteria)
        df0 = (signals.index.values <= loc) & ((loc<signals['t1'])| pd.isnull(signals['t1'])) 
        # return the index of those trades that make this trade active (poor english)
        act = signals[df0].index
        # if is active, take the mean
        if len(act) > 0:
            out[loc] = signals.loc[act,'signal'].mean()
        # if not, assign 0, meaning no signals active at this time
        else:
            out[loc] = 0 # no signals active at this time
    return out

def discreteSignal(signal0, stepSize):
    """
    discrete signal
    : param signals: a pd.df with
                Index: time, the start of the signal
                'signal': the betsize (averaged)
    : param stepSize: constant (0,1], degree of discretization

    : return: signals with discretised 'signal'
    """
    signal1 = (signal0/stepSize).round() * stepSize # discretize
    signal1[signal1>1] = 1 # cap, cannot > than 1
    signal1[signal1<-1] = -1 # floor cannot < than 1
    return signal1

#####################################################################################
#                       Dynamic Position Size and Limit Price
#####################################################################################

def betSize(w,x):
    """
    : params w: float, width of the sigmoid function 
    : params x: float, (f_i - p_t), the divergence between the forecast and the current market price
    
    : return: float, bet size
    """
    return x*(w+x**2)**-.5
# ----------------------------------------------------------------------------------

def getW(x,m):
    """
    The inverse function of the betSize with respect to the width of the sigmoid function w
    : params x: float, (f_i - p_t), the divergence between the forecast and the current market price
    : params m: float bet size

    : return: float, width of the sigmoid function w
    """
    # 0 < alpha < 1
    return x**2*(m**-2-1)
# ----------------------------------------------------------------------------------

def getTPos(w,f,mP,maxPos):
    """
    Get the target position size associated with the forecast f
    : params w: float, width of the sigmoid function
    : params f: float, forecast price
    : params mP: float, market price
    : params maxPos: maximum absolute position size
    
    : return: float, target position size
    """
    return int(betSize(w,f-mP)*maxPos)
# ----------------------------------------------------------------------------------
def invPrice(f,w,m):
    """
    The inverse function of the betSize with respect to the market price P
    : params f: float, forecast price
    : params w: float, width of the sigmoid function
    : params m: float, the bet size

    : return: float, the inverse price:
    """
    return f-m*(w/(1-m**2))**.5
# ----------------------------------------------------------------------------------
def limitPrice(tPos,pos,f,w,maxPos):
    """
    Get the breakeven limit price to realise the gain or avoid loss
    : params tPos: float, target position size
    : params pos: float, current position size
    : params f: float, forecast price
    : params w: width of the sigmoid function
    : params maxPos: maximum absolute position size
    
    : return: float, limit price"""
    sgn = (1 if tPos >= pos else -1)
    lP = 0
    for j in range(abs(pos+sgn), abs(tPos+1)): # xrange() -> range() Py3
        lP += invPrice(f,w,j/float(maxPos))
    lP /= tPos - pos
    return lP
# ----------------------------------------------------------------------------------

def main():
    # test code
    pos, maxPos, mP, f, wParams = 0, 100, 100, 115, {'divergence':10, 'm':.95}
    # 1) calibrate the sigmoid function
    w = getW(wParams['divergence'],wParams['m']) # calibrate w
    # 2) compute the target position
    tPos = getTPos(w,f,mP,maxPos) # get tPos
    # 3) find the limit price
    lP = limitPrice(tPos, pos, f, w, maxPos) # limit price for order
    print(tPos)
    print(lP)
    return
# ----------------------------------------------------------------------------------
if __name__ == '__main__': main()