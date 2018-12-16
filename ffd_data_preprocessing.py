from pathlib import PurePath, Path
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
import platform
from mpEngine import linParts, expandCall, reportProgress
from ffd import optimal_ffd, fracDiff_FFD
import time
pp = PurePath(Path.cwd()).parts[:]
pdir = PurePath(*pp)
data_dir = pdir / 'data'
rawDataDir = PurePath(data_dir / 'interim3' / 'STI')
interimDataDir = PurePath(data_dir / 'interimFFD' / 'STI')

def mpOptimalFFD(molecules, rawDataDir = rawDataDir, interimDataDir = interimDataDir):
    for equity in molecules:
        infp = PurePath(str(rawDataDir) + "/" + equity)
        vbar = pd.read_parquet(infp)
        vbar = vbar.replace([np.inf,-np.inf],np.nan)
        vbar = vbar.dropna()
        ## real stuff
        print(equity)
        logP = np.log(vbar.price)
        logP = logP[~logP.index.duplicated()]
        min_ffd = optimal_ffd(logP)
        dfx2 = fracDiff_FFD(logP.to_frame(),min_ffd)

        ## output
        tmpPath = str(interimDataDir) + "/" + equity
        outfp = PurePath(tmpPath)
        print(outfp)
        dfx2.to_parquet(outfp)
        print(equity, "--Success: save")
    return

def main():

    equities = os.listdir(rawDataDir)
    
    if platform.system() == 'Windows':
        cpus = 1
    else:
        cpus = mp.cpu_count()

    parts = linParts(len(equities), cpus)
    
    func = mpOptimalFFD
    jobs = []
    for i in range(1, len(parts)):
        job = {'molecules': equities[parts[i - 1] : parts[i]], 'func': func}
        jobs.append(job)

    task = jobs[0]['func'].__name__
    time0 = time.time()
    pool = mp.Pool(processes = cpus) # i7 I cores..should delete 'numThreads' really
    # 'map': map the function to the arguments/parameters
    # 'pool.map': parallelise `expandCall`
    # 'imap_unordered`: iterators, results will be yielded as soon as they are ready, regardless of the order of the input iterable
    outputs = pool.imap_unordered(expandCall, jobs) # 'imap_unordered` seems to use less memory than 'imap'
    
    # Process asyn output, report progress
    # I guess the results are actually output here
    for i, out_ in enumerate(outputs, 1): # index start at 1
        reportProgress(i, len(jobs), time0, task)
    pool.close() # close the pool, stop accepting new jobs
    pool.join() # this is needed to prevent memory leaks
    
    return

if __name__ == "__main__":
    main()