import pandas as pd
import numpy as np
from numba import jit
import datetime as dt
import time
import sys
import copyreg,types, multiprocessing as mp
import copy
import platform
from multiprocessing import cpu_count

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)
#________________________________
def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj,cls)

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


#===============================================================================================================
#           Performance
#=================================================================================================================
def linParts(numAtoms, numThreads):
    # partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1) # find the indices (may not int) of the partition parts
    parts = np.ceil(parts).astype(int) # ceil the float indices into int
    return parts

def nestedParts(numAtoms, numThreads, upperTriang = False):
    # partition of atoms with an inner loop
    parts = [0]
    numThreads_ = min(numThreads,numAtoms) 
    for _ in range(numThreads_):
        # find the appropriate size of each part by an algorithms
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + numAtoms * (numAtoms + 1.) / numThreads_)
        part = (-1 + part**.5) / 2.
        # store part into parts
        parts.append(part)
    # rounded to the nearest natural number
    parts = np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts = np.cumsum(np.diff(parts)[ : :-1])
        # dont forget the 0 at the begining
        parts = np.append(np.array([0]), parts)
    return parts


def mpPandasObj(func, pdObj, numThreads = 24, mpBatches = 1, linMols = True, **kargs):
    '''
    Parallelize jobs, return a dataframe or series
    :params func: function to be parallelized. Returns a DataFrame
    :params pdObj: tuple,
        + pdObj[0]: Name of argument used to pass the molecule
        + pdObj[1]: List of atoms that will be grouped into molecules
    :params numThreads: int, no. of threads that will be used in parallel (1 processor per thread)
    :params mpBatches: int, no. of parallel batches (jobs per core)
    :params linMols: bool, whether partitions will be linear or double-nested
    :params kwds: any other argument needed by func

    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''
    import pandas as pd
    # ----------------Partition the dataset-------------------------
    # parts: the indices to separate
    if linMols:
        parts = linParts(len(pdObj[1]), numThreads * mpBatches)
    else:
        parts = nestedParts(len(pdObj[1]), numThreads * mpBatches)

    jobs = []
    for i in range(1, len(parts)):
        # name of argument: molecule, function: func
        job = {pdObj[0]: pdObj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs) # update kargs?
        jobs.append(job)
    # -----------------multiprocessing--------------------
    if numThreads == 1:
        out = processJobs_(jobs)
    else:
        out = processJobs(jobs, numThreads = numThreads)
    #------------determine the datatype of the output----
    try:
        if len(out) == 0:
            return pd.DataFrame()
        elif isinstance(out[0], pd.DataFrame):
            df0 = pd.DataFrame()
        elif isinstance(out[0], pd.Series):
            df0 = pd.Series()
        else:
            return out
        # Append the output to the df0
        for i in out:
            df0 = df0.append(i)
        # sort objects by labels
        df0 = df0.sort_index()
    except:
        print(type(out))
        df0 = pd.DataFrame()
    return df0

def processJobs_(jobs):
    # Run jobs sequentially, for debugging or numThread = 1
    out=[]
    for job in jobs:
        out_ = expandCall(job)
        out.append(out_)
    return out
#________________________________
def reportProgress(jobNum, numJobs, time0, task):
    # Report progress as asynch jobs are completed
    # keep us informed about the percentage of jobs completed
    # msg[0]: % completed, msg[1]: time elapses
    msg = [float(jobNum) / numJobs, (time.time() - time0)/60.]
    # msg[2]:minutes remaining
    msg.append(msg[1] * (1 / msg[0] - 1))
    # the current time
    timeStamp = str(dt.datetime.fromtimestamp(time.time()))
    # convert a list `msg` into a string `msg`
    msg = timeStamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
        str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'
    
    if jobNum < numJobs: 
        sys.stderr.write(msg+'\r') # pointer goes to the front?
    else:
        sys.stderr.write(msg+'\n') # pointer goes to the next line
    return
#________________________________
def processJobs(jobs, task = None, numThreads = 24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None: 
        task = jobs[0]['func'].__name__
    pool = mp.Pool(processes = numThreads) # i7 I cores..should delete 'numThreads' really
    # 'map': map the function to the arguments/parameters
    # 'pool.map': parallelise `expandCall`
    # 'imap_unordered`: iterators, results will be yielded as soon as they are ready, regardless of the order of the input iterable
    outputs = pool.imap_unordered(expandCall, jobs) # 'imap_unordered` seems to use less memory than 'imap'
    out = []
    time0 = time.time()
    # Process asyn output, report progress
    # I guess the results are actually output here
    for i, out_ in enumerate(outputs, 1): # index start at 1
        out.append(out_)
        reportProgress(i, len(jobs), time0, task)
    pool.close() # close the pool, stop accepting new jobs
    pool.join() # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    # Unwrap the items(atoms) in the job(molecule) and execute the callback function
    func = kargs['func'] # function
    del kargs['func'] # delete the `function` column/argument
    out = func(**kargs) # put the arguments into the function
    return out

def processJobsRedux(jobs, task = None, cpus = 4, redux = None, reduxArgs = {}, reduxInPlace = False):
    '''
    Run in parallel
    jobs must contain a ’func’ callback, for expandCall
    redux prevents wasting memory by reducing output on the fly
    :params redux: func, a callback to the function that carries out the reduction, e.g. pd.DataFrame.add
    :params reduxArgs: dict, contains the keyword arguments that must be passed to the redux (if any)
        e.g. if redux = 'od,DataFrame.join, reduxArg = {'how':'outer'}
    :params reduxInPlace: bool, indicate whether the redux operation should happen in-place or not
        e.g. redux = dict.update or redux = list.append requires reduxInplace = True 
            because updating a dictionary or appending a list is both in-place operations
    '''
    
    
    if task is None: # get the name of the function/tasl
        task = jobs[0]['func'].__name__
    # 'map': map the function to the arguments/parameters
    # 'pool.map': parallelise `expandCall`
    # 'imap_unordered`: iterators, results will be yielded as soon as they are ready, regardless of the order of the input iterable
    pool = mp.Pool(processes = cpus)
    imap = pool.imap_unordered(expandCall, jobs)
    out = None
    time0 = time.time()
    # Process asynchronous output, report progress
    for i, out_ in enumerate(imap, 1):
        if out is None: # the first element
            if redux is None: # if the reduction function is not specified
                out = [out_]
                redux = list.append
                reduxInPlace = True
            else: 
                out = copy.deepcopy(out_)
        else: # not the first
            if reduxInPlace: # if inplace, no need to re-assign to out
                redux(out, out_, **reduxArgs)
            else:
                out = redux(out, out_, **reduxArgs)
        reportProgress(i, len(jobs), time0, task)
    pool.close() # close the pool, stop accepting new jobs
    pool.join() # this is needed to prevent memory leaks
    if isinstance(out, (pd.Series, pd.DataFrame)):
        out = out.sort_index()
    return out

def mpJobList(func, argList, numThreads, mpBatches = 1,  linMols = True, redux = None, reduxArgs ={} , reduxInPlace = False, **kargs):
    '''
    Parallelize jobs, return a dataframe or series
    :params func: function to be parallelized. Returns a DataFrame
    :params argList: tuple,
        + argList[0]: Name of argument used to pass the molecule
        + argList[1]: List of atoms that will be grouped into molecules
    :params mpBatches: int, no. of parallel batches (jobs per core)
    :params linMols: bool, whether partitions will be linear or double-nested
    :params redux: func, a callback to the function that carries out the reduction, e.g. pd.DataFrame.add
    :params reduxArgs: dict, contains the keyword arguments that must be passed to the redux (if any)
        e.g. if redux = 'od,DataFrame.join, reduxArg = {'how':'outer'}
    :params reduxInPlace: bool, indicate whether the redux operation should happen in-place or not
        e.g. redux = dict.update or redux = list.append requires reduxInplace = True 
            because updating a dictionary or appending a list is both in-place operations

    Example: df1=mpJobList(func,('molecule',df0.index),24)
    '''

    # ----------------Partition the dataset-------------------------
    # parts: the indices to separate
    if numThreads:
        cpus = numThreads
    else:
        if platform.system() == 'Windows':
            cpus = 1
        else:
            cpus = cpu_count() - 1

    if linMols:
        parts = linParts(len(argList[1]), cpus * mpBatches)
    else:
        parts = nestedParts(len(argList[1]), cpus * mpBatches)
    jobs = []

    for i in range(1, len(parts)):
        job = {argList[0]: argList[1][parts[i - 1] : parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)
    # -----------------multiprocessing--------------------
    out = processJobsRedux(jobs, redux = redux, reduxArgs = reduxArgs,
                         reduxInPlace = reduxInPlace, cpus = cpus)
    # no need to process an outputed list, save memory and time
    return out