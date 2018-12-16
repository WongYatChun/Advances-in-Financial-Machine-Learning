from pathlib import PurePath, Path
import os
import sys
import pandas as pd
from bars import read_bbg_ticks

pp = PurePath(Path.cwd()).parts[:]
pdir = PurePath(*pp)
data_dir = pdir / 'data'

RANDOM_STATE = 777
rawDataDir = PurePath(data_dir/'raw'/'ASX')
interimDataDir = PurePath(data_dir/'interim'/'ASX')

for equity in os.listdir(rawDataDir):
    path = PurePath(str(rawDataDir) + "/" + equity)
    print(path)
    df = read_bbg_ticks(path)
    print("Success: Df")
    tmpPath = str(interimDataDir) + "/" + equity.replace(' ','_').replace('csv','parq')
    outfp = PurePath(tmpPath)
    print(outfp)
    df.to_parquet(outfp)
    print("Success: save")
