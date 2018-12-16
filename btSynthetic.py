import numpy as np
from random import gauss
from itertools import product
#———————————————————————————————————————
# PYTHON CODE FOR THE DETERMINATION OF OPTIMAL TRADING RULES
def main():
    rPT = rSLm = np.linspace(0, 10, 21) # construct a mesh of profit-taking and stop-loss pairs
    for prod_ in product([10, 5, 0, -5, -10], [5, 10, 25, 50, 100]): 
        coeffs = {'forecast': prod_[0], 'hl': prod_[1], 'sigma': 1} # coefficients: 'forecast' = E_0[P_T]
        output = batch(coeffs, nIter = 1e5, maxHP = 100, rPT = rPT,rSLm = rSLm) # mean, std, and SR of each pair
    return output

def batch(coeffs, nIter = 1e5, maxHP = 100, rPT = np.linspace(.5, 10, 20), 
          rSLm = np.linspace(.5, 10, 20), seed = 0):
    """
    compute the Sharpe ratios associated with various trading rules
    suppose a discrete O-U process on prices
    the half-life of the process: hl = -log(2)/log(phi)
    """
    # phi: the speed at which the starting P_0 converged toward E_0[P_T]
    # find phi given half life
    phi = 2**(-1. / coeffs['hl'])
    # empty list for the output
    output1 = []
    # for each (profit_taking, stop_loss) pair
    for comb_ in product(rPT, rSLm):
        output2 = [] # empty list
        for _ in range(int(nIter)): # each iteration/paths
            p = seed 
            hp = 0
            while True:
                # discrete O-U process on the price
                p = (1 - phi) * coeffs['forecast'] + phi * p + coeffs['sigma'] * gauss(0, 1)
                # price change (return) from the new P to the initial price
                cP = p - seed
                # horizon + 1
                hp += 1
                # if price change is larger than the profit_taking or less than the stop_lossing threshold or the maxHorizon is reached
                if cP > comb_[0] or cP < -comb_[1] or hp > maxHP: 
                    output2.append(cP) # add the price change into the output2 array
                    break
        mean = np.mean(output2) # find the mean of the returns of this pair of Profit_taking and Stop-loss threshold
        std = np.std(output2) # find the std of the returns of this pair of Profit_taking and Stop-loss threshold
        print (comb_[0], comb_[1], mean, std, mean / std) # print the values: rPT, rSLm, mean, std, and Shape ratio
        output1.append((comb_[0], comb_[1], mean,std, mean/std)) # store them into the output1
    return output1