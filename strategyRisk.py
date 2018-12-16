import numpy as np,scipy.stats as ss
# from sympy import *
# init_printing(use_unicode=False,wrap_line=False,no_global=True)
# p,u,d=symbols('p u d')

def binSR(sl,pt,freq,tSR):
    """
    Given a trading rule characterized by the parameters {sl,pt,freq},
    what's the min precision p required to achieve a Sharpe ratio tSR?
    1) Inputs
    sl: stop loss threshold
    pt: profit taking threshold
    freq: number of bets per year
    tSR: target annual Sharpe ratio
    2) Output
    p: the min precision rate p required to achieve tSR
    """
    a = (freq + tSR**2) * (pt - sl)**2
    b = (2 * freq * sl - tSR**2 * (pt - sl)) * (pt - sl)
    c = freq * sl**2
    p = (-b + (b**2 - 4 * a * c)**.5) / (2. * a)
    return p

def binFreq(sl,pt,p,tSR):
    """
    Given a trading rule characterized by the parameters {sl,pt,freq},
    what's the number of bets/year needed to achieve a Sharpe ratio
    tSR with precision rate p?
    Note: Equation with radicals, check for extraneous solution.
    1) Inputs
    sl: stop loss threshold
    pt: profit taking threshold
    p: precision rate p
    tSR: target annual Sharpe ratio
    2) Output
    freq: number of bets per year needed
    """
    freq = (tSR * (pt - sl))**2 * p * (1 - p) / ((pt - sl) * p + sl)**2 # possible extraneous
    if not np.isclose(binSR(sl,pt,freq,p),tSR): # CHECK
        return
    return freq
#———————————————————————————————————————
def mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs):
    # Random draws from a mixture of gaussians
    ret1 = np.random.normal(mu1, sigma1, size = int(nObs * prob1))
    ret2 = np.random.normal(mu2, sigma2, size = int(nObs) - ret1.shape[0])
    ret = np.append(ret1, ret2, axis = 0)
    np.random.shuffle(ret)
    return ret
#———————————————————————————————————————
def probFailure(ret, freq, tSR):
    # Derive probability that strategy may fail
    rPos = ret[ret > 0].mean() # mean pos return
    rNeg = ret[ret <= 0].mean() # mean neg return
    p = ret[ret > 0].shape[0] / float(ret.shape[0]) # prob. of pos return (precision rate)
    thresP = binSR(rNeg, rPos, freq, tSR) # calculate the threshold P for given SR
    risk = ss.norm.cdf(thresP, p, p * (1 - p)) # approximation to bootstrap
    return risk
#———————————————————————————————————————
def main():
    #1) Parameters
    mu1, mu2, sigma1, sigma2, prob1, nObs = .05, -.1, .05, .1, .75, 2600
    tSR, freq = 2., 260
    #2) Generate sample from mixture
    ret = mixGaussians(mu1, mu2, sigma1, sigma2, prob1, nObs)
    #3) Compute prob failure
    probF = probFailure(ret, freq, tSR)
    print ('Prob strategy will fail', probF)
    return
#———————————————————————————————————————
if __name__=='__main__':main()