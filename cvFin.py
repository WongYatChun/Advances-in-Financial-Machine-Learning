import pandas as pd
import numpy as np 

from sklearn.model_selection._split import _BaseKFold

def getTrainTimes(t1,testTimes):
    """
    Given testTimes, find the times of the training observations.
    Purge from the training set all observations whose labels overlapped in time with those labels included in the testing set
    :params t1: event timestamps
        —t1.index: Time when the observation started.
        —t1.value: Time when the observation ended.
    :params testTimes: pd.series, Times of testing observations.
    :return trn: pd.df, purged training set
    """
    # copy t1 to trn
    trn = t1.copy(deep = True)
    # for every times of testing obervation
    for i, j in testTimes.iteritems():
        # cond 1: train starts within test
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index 
        # cond 2: train ends within test
        df1 = trn[(i <= trn) & (trn <= j)].index 
        # cond 3: train envelops test
        df2 = trn[(trn.index <= i) & (j <= trn)].index 
        # drop the data that satisfy cond 1 & 2 & 3
        trn = trn.drop(df0.union(df1).union(df2))
    return trn

# def getEmbargoTimes(times,pctEmbargo):
#     """ Not sure if it works
#     # Get embargo time for each bar
#     :params times: time bars
#     :params pctEmbargo: float, % of the bars will be embargoed
#     :return trn: pd.df, purged training set
#     """
#     # cal no. of steps from the test data
#     step=int(times.shape[0]*pctEmbargo)
#     if step == 0:
#         # if no embargo, the same data set
#         mbrg=pd.Series(times,index=times)
#     else:
#         #
#         mbrg=pd.Series(times[step:],index=times[:-step])
#         mbrg=mbrg.append(pd.Series(times[-1],index=times[-step:]))
#     return mbrg
# #———————————————————————————————————————
    # testTimes=pd.Series(mbrg[dt1],index=[dt0]) # include embargo before purge
    # trainTimes=getTrainTimes(t1,testTimes)
    # testTimes=t1.loc[dt0:dt1].index

class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between
    """
    def __init__(self, n_splits = 3, t1 = None, pctEmbargo = 0.):
        if not isinstance(t1, pd.Series):
            # if t1 is not a pd.series, raise error
            raise ValueError('Label Through Dates must be a pd.Series')
        # inherit _BaseKFold, no shuffle
        # Might be python 2x style
        super(PurgedKFold, self).__init__(n_splits, shuffle = False, random_state = None)
        self.t1 = t1 # specify the vertical barrier
        self.pctEmbargo = pctEmbargo # specify the embargo parameter (% of the bars)

    def split(self, X, y = None, groups = None):
        """
        :param X: the regressors, features
        :param y: the regressands, labels
        :param groups: None

        : return
            + train_indices: generator, the indices of training dataset 
            + test_indices: generator, the indices of the testing dataset
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            # X's index does not match t1's index, raise error
            raise ValueError('X and ThruDateValues must have the same index')
        # create an array from 0 to (X.shape[0]-1)
        indices = np.arange(X.shape[0])
        # the size of the embargo
        mbrg = int(X.shape[0] * self.pctEmbargo)
        # list comprehension, find the (first date, the last date + 1) of each split
        test_starts = [(i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i,j in test_starts: # for each split
            t0 = self.t1.index[i] # start of test set
            test_indices = indices[i : j] # test indices are all the indices from i to j
            maxT1Idx = self.t1.index.searchsorted(self.t1[test_indices].max()) # find the max(furthest) vertical barrier among the test dates
            # index.searchsorted: find indices where element should be inserted (behind) to maintain the order
            # find all t1.indices (the start dates of the event) when t1.value (end date) < t0
            # i.e the left side of the training data
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index) 
            if maxT1Idx < X.shape[0]: # right train (with embargo)
                # indices[maxT1Idx+mbrg:]: the indices that is after the (maxTestDate + embargo period) [right training set]
                # concat the left training indices and the right training indices
                train_indices = np.concatenate((train_indices, indices[maxT1Idx + mbrg: ]))
        # the function return generators for the indices of training dataset and the indices of the testing dataset respectively
        yield train_indices, test_indices

def cvScore(clf, X, y, sample_weight, scoring = 'neg_log_loss', t1 = None, cv = None, cvGen = None, pctEmbargo = 0):
    """
    Address two sklearn bugs
    1) Scoring functions do not know classes_
    2) cross_val_score will give different results because it weights to the fit method, but not to the log_loss method
    
    :params pctEmbargo: float, % of the bars will be embargoed
    """
    if scoring not in ['neg_log_loss', 'accuracy']:
        # if not using 'neg_log_loss' or 'accuracy' to score, raise error
        raise Exception('wrong scoring method.')
    from sklearn.metrics import log_loss,accuracy_score # import log_loss and accuracy_score
    #   from clfSequential import PurgedKFold # the original code assume they are stored in different folder
    if cvGen is None: # if there is no predetermined splits of the test sets and the training sets
        # use the PurgedKFold to generate splits of the test sets and the training sets
        cvGen = PurgedKFold(n_splits = cv,t1 = t1,pctEmbargo = pctEmbargo) # purged
    score = [] # store the CV scores
    # for each fold
    for train,test in cvGen.split(X = X):
        # fit the model
        fit = clf.fit(X = X.iloc[train, : ], y = y.iloc[train], sample_weight = sample_weight.iloc[train].values)
        if scoring == 'neg_log_loss':
            prob = fit.predict_proba(X.iloc[test, : ]) # predict the probabily
            # neg log loss to evaluate the score
            score_ = -1 * log_loss(y.iloc[test], prob, sample_weight = sample_weight.iloc[test].values, labels = clf.classes_)
        else:
            pred = fit.predict(X.iloc[test, : ]) # predict the label
            # predict the accuracy score
            score_ = accuracy_score(y.iloc[test], pred, sample_weight = sample_weight.iloc[test].values)
        score.append(score_)
    return np.array(score)