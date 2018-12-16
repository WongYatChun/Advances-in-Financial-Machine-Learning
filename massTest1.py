from pathlib import PurePath, Path
import sys
import os
pp = PurePath(Path.cwd()).parts[:]
pdir = PurePath(*pp)
data_script_dir = pdir / 'src' / 'data'
bars_script_dir = pdir / 'src' / 'features'
sys.path.append(data_script_dir.as_posix())
sys.path.append(bars_script_dir.as_posix())
data_dir = pdir / 'data'
from multiprocessing import cpu_count
# import python scientific stack
import pandas as pd
import numpy as np
import platform
# import util libs
from labelling import getBins, getEvents, getTEvents, dropLabels, addVerticalBarrier
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans


oriDataDir = PurePath(data_dir / 'interim3' / 'ASX')
pcaDataDir = PurePath(data_dir / 'interim4' / 'ASX')

result = pd.DataFrame(columns=['equity', 'No. of Trade', 'Total Return',
                  'Mean Return', 'Standardized Deviation',
                  'Win from Long', 'Win from Short'])

ffdDataDir = PurePath(data_dir / 'interimFFD' / 'STI')
oriDataDir = PurePath(data_dir / 'interim3' / 'STI')
pcaDataDir = PurePath(data_dir / 'interim4' / 'STI')
for equity in os.listdir(pcaDataDir):
    #ffd
    #df_ffd = pd.read_parquet(PurePath(str(ffdDataDir) + "/" + equity))
    # ori
    df_original = pd.read_parquet(PurePath(str(oriDataDir) + "/" + equity))
    # pca
    df_PCA = pd.read_parquet(PurePath(str(pcaDataDir) + "/" + equity))
    
    
    n_components = 2
    #hmmmodel = GaussianHMM(n_components = n_components)
    #hmmmodel.fit(df_ffd[['price']])
    #df_ffd['hmm_fracdiff'] = hmmmodel.predict(df_ffd[['price']])

    #df_hmm = df_original.iloc[:,10:].join(df_ffd['hmm_fracdiff'], how = 'left')
    df_hmm = df_original.iloc[:,10:].join(df_PCA, how = 'left').dropna()
    n_clusters = 2
    kmean = KMeans(init = 'k-means++', n_clusters = n_clusters)
    df_hmm['hmm_predict'] = kmean.fit_predict(df_hmm)
    df_hmm['hmm_predict'] = [1 if x == 0 else x for x in df_hmm['hmm_predict'] ]
    print(df_hmm.shape[0])
       # tEvents are the events where there is a accumulative forecast error
    tEvents = getTEvents(df_original['forecasts_error'], h = df_original['error_std'], symmetric = True, isReturn = True) # isReturn is awkward
    
    df_original['side'] = [np.sign(x) if abs(x) > 0 else np.nan for x in df_original['normOI'] ]
    df_original['side'] = - df_original['side'] * df_hmm['hmm_predict']
    # find the vertical barrier
    t1 = addVerticalBarrier(tEvents, df_original['price'])
    print(t1.shape)
    # profit and loss threshold
    ptsl = [1,1]

    minRet = 0 #df_original['dailyVol'].mean()

    # Run in single-threaded mode on Windows
    if platform.system() == "Windows":
        cpus = 1
    else:
        cpus = cpu_count() - 1

    # get trading events
    events = getEvents(df_original['price'], tEvents, ptsl, df_original['dailyVol'], minRet,
                       cpus, t1=t1, side = df_original['side'])
    #events = getEvents(vbar['price'],tEvents,ptsl,vbar['dailyVol'],minRet,cpus,t1=t1)
    # events = getEvents(vbar['price'],tEvents,ptsl,abs(vbar['normOI']),minRet*2,cpus,t1=t1)
    KF_bins = getBins(events,df_original['price']).dropna()
#    KF_bins = dropLabels(KF_bins)
    combined = KF_bins.join(df_original['side'], how = "left").dropna()
    
    result.append({'equity': equity.replace('_parq', ''), 
                   'No. of Trade': combined.shape[0],
                   'Total Return': combined.ret.sum(),
                   'Mean Return': combined.ret.sum()/combined.shape[0],
                   'Standardized Deviation': combined.ret.std(),
                   'Win from Long': combined[(combined['bin'] == 1) & (combined['side'] == 1)].shape[0],
                   'Win from Short': combined[(combined['bin'] == 1) & (combined['side'] == -1)].shape[0]}, 
                    ignore_index=True)
    print(equity)
    print(KF_bins.bin.value_counts())
    print("Total Return: %f%%" % (100 * KF_bins.ret.sum()))
    print("Mean Return: %f%%" % (100 * KF_bins.ret.sum()/KF_bins.shape[0]))
result.to_excel("ASX_result.xlsx")