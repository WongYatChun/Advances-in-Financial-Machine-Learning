from pathlib import PurePath, Path
import os
import sys
import pandas as pd
import numpy as np
from bars import read_bbg_ticks
import scipy.stats as stats
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from utils import getDailyVol
from bars import volume_bar_df

pp = PurePath(Path.cwd()).parts[:]
pdir = PurePath(*pp)
data_dir = pdir / 'data'

def orderFlow(df, degreeFree=9999999, vpinSpan=50):
    # cal normalised order imbalance
    df['normOI'] = 2 * stats.t.cdf(df['retClose'] / df['dailyVol'], degreeFree) - 1
    # cal VPIN
    df['VPIN'] = df['normOI'].abs().rolling(vpinSpan).mean()
    return df


def df_rolling_autocorr(df, window, lag=1):
    """Compute rolling column-wise autocorrelation for a DataFrame."""

    return (df.rolling(window=window).corr(df.shift(lag)))  # could .dropna() here


rawDataDir = PurePath(data_dir / 'interim' / 'NKY')
interimDataDir = PurePath(data_dir / 'interim2' / 'NKY')
def main1():
    for equity in os.listdir(rawDataDir):
        infp = PurePath(str(rawDataDir) + "/" + equity)
        df = pd.read_parquet(infp)
        volume_M = df.volume.sum() / df.shape[0]
        # produce the volume bar
        vbar = volume_bar_df(df, 'volume', volume_M)
        vbar.set_index('dates', inplace=True)
        # return
        vbar['retClose'] = vbar['price'] / vbar['price'].shift(1) - 1
        # daily vol
        vbar['dailyVol'] = getDailyVol(vbar['price'])

        # normOI and VPIN
        vbar = orderFlow(vbar)
        # kf setting, assume random walk
        kf = KalmanFilter(1, 1)
        sigma_h = 0.0001  # hidden
        sigma_e = 0.001  # obs
        kf.obs_cov = np.array([sigma_e])
        kf.state_cov = np.array([sigma_h])
        kf.design = np.array([1.0])
        kf.transition = np.array([1.0])
        kf.selection = np.array([1.0])
        kf.initialize_known(np.array([vbar.price[0]]), np.array([[sigma_h]]))
        kf.bind(np.array(vbar.price.copy()))
        r = kf.filter()
        vbar['forecasts'] = pd.DataFrame(r.forecasts[0], index=vbar.index)
        vbar['forecasts_error'] = pd.DataFrame(r.forecasts_error[0], index=vbar.index)
        vbar['error_std'] = pd.DataFrame(np.sqrt(r.forecasts_error_cov[0][0]), index=vbar.index)
        vbar = vbar.dropna()
        # srl_corr
        vbar['srl_corr'] = df_rolling_autocorr(vbar['price'], window=100).rename('srl_corr')
        vbar = vbar.dropna()

        ## output
        tmpPath = str(interimDataDir) + "/" + equity
        outfp = PurePath(tmpPath)
        print(outfp)
        vbar.to_parquet(outfp)
        print("Success: save")
    return

rawDataDir = PurePath(data_dir / 'interim2' / 'ASX')
interimDataDir = PurePath(data_dir / 'interim3' / 'ASX')

from hmmlearn.hmm import GaussianHMM
n_components = 2
hmmmodel = GaussianHMM(n_components = n_components)
def main2_hmm():
    for equity in os.listdir(rawDataDir):
        infp = PurePath(str(rawDataDir) + "/" + equity)
        vbar = pd.read_parquet(infp)
        vbar = vbar.replace([np.inf,-np.inf],np.nan)
        vbar = vbar.dropna()
        ## real stuff


        #hmmmodel.fit(np.log(vbar[['price']]))
        #vbar['hmm_logprice'] = hmmmodel.predict(np.log(vbar[['price']]))

        hmmmodel.fit(vbar[['retClose']])
        vbar['hmm_retClose'] = hmmmodel.predict(vbar[['retClose']])

        hmmmodel.fit(vbar[['dailyVol']])
        vbar['hmm_dailyVol'] = hmmmodel.predict(vbar[['dailyVol']])

        hmmmodel.fit(vbar[['VPIN']])
        vbar['hmm_VPIN'] = hmmmodel.predict(vbar[['VPIN']])

        hmmmodel.fit(vbar[['normOI']])
        vbar['hmm_normOI'] = hmmmodel.predict(vbar[['normOI']])

        hmmmodel.fit(vbar[['srl_corr']])
        vbar['hmm_srl_corr'] = hmmmodel.predict(vbar[['srl_corr']]) 
        
        ## output
        tmpPath = str(interimDataDir) + "/" + equity
        outfp = PurePath(tmpPath)
        print(outfp)
        vbar.to_parquet(outfp)
        print("Success: save")
    return