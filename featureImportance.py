import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,accuracy_score
from cvFin import PurgedKFold, cvScore
from itertools import product
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from mpEngine import mpPandasObj


def featImpMDI(fit, featNames):
    """
    feat importance based on In-Sample (IS) mean impurity reduction
    :params fit: fit model
    :params featNames: str, the name of the features
    :return imp:
    """
    # find the feature importance of each estimator
    df0 = {i : tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    # convert df from dicts, the keys are rows
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    # the column names are the feature names
    df0.columns = featNames
    # make sure features with 0 importance are not averaged
    # since the only reason for a 0 is that the feature is not randomly chosen
    df0 = df0.replace(0, np.nan) # because max_features=1
    # imp is a dict that document ??
    imp = pd.concat({'mean': df0.mean(), 'std': df0.std() * df0.shape[0]**-.5}, axis=1)
    # feature importances add up to 1 and each is bounded between 0 and 1
    # but it is strange
    imp /= imp['mean'].sum()
    return imp

def featImpMDA(clf, X, y, cv, sample_weight, t1, pctEmbargo, scoring = 'neg_log_loss'):
    """
    feature importance based on OOS score reduction
    1) Fit a classifier
    2) derive its OOS performance (neg_log_loss or accuracy)
    3) permutate each column of the featurs matrix (X), one at a time
    4) Derive the OOS performance after the permutation
    5) the worse the performance after, the more important the feature
    :params clf:
    :params X:
    :params y:
    :params cv:
    :params sample_weight:
    :params t1:
    :params pctEmbargo:
    :params scoring:
    
    :return 
        imp
        scr0.mean() 
    """
    # if the CV scoring scheme is neither 'neg_log_loss' or 'accuracy'
    if scoring not in ['neg_log_loss', 'accuracy']:
        raise Exception('wrong scoring method.') # raise error/exception

    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv with embargo

    scr0 = pd.Series()  # empty pd.series
    scr1 = pd.DataFrame(columns = X.columns) # empty pd.df, but columns are the feature names (X,columns)

    # may be shortened using modole cvFIn.py
    for i, (train, test) in enumerate(cvGen.split(X = X)): # for each CV fold
        # x0, y0, and w0 are the training features, training labels, and the sample weights for the training data respectively
        X0, y0, w0 = X.iloc[train, : ], y.iloc[train], sample_weight.iloc[train]
        # x1, y1, and w1 are the testing features, testing labels, and the sample weights for the testing data respectively
        X1, y1, w1 = X.iloc[test, : ], y.iloc[test], sample_weight.iloc[test]
        # fit the training set to the classifier
        fit = clf.fit(X = X0, y = y0, sample_weight = w0.values)
        if scoring == 'neg_log_loss': # if the scoring is "neg_log_loss"
            prob = fit.predict_proba(X1) # predict the prob
            scr0.loc[i] = -1 * log_loss(y1,prob, sample_weight = w1.values, labels = clf.classes_) # cal the neg_log_loss
        else: # if the scoring is the "accuracy"
            pred = fit.predict(X1) # predict label
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight = w1.values) # produce the accuracy score
        for j in X.columns: # for each feature
            X1_ = X1.copy(deep = True) # X1_ = X1
            np.random.shuffle(X1_[j].values) # permutation of a single column
            if scoring == 'neg_log_loss': # if the scoring is "neg_log_loss"
                prob = fit.predict_proba(X1_) # predict the prob
                scr1.loc[i,j] = -1*log_loss(y1, prob, sample_weight = w1.values, labels = clf.classes_) # cal the neg_log_loss
            else: # if the scoring is the "accuracy"
                pred = fit.predict(X1_) # predict label
                scr1.loc[i,j] = accuracy_score(y1, pred, sample_weight = w1.values) # produce the accuracy score
    # add -scr0 to scr1, axis to match Series index on 'index'
    imp = (-scr1).add(scr0, axis = 0)
    if scoring == 'neg_log_loss': # if the scoring is "neg_log_loss"
        imp = imp / -scr1
    else: # if the scoring is the "accuracy"
        imp = imp / (1. - scr1)
    # cal mean feature importance, std of mean importance
    imp = pd.concat({'mean': imp.mean(), 'std': imp.std() * imp.shape[0]**-.5}, axis=1)
    # return feature importance indicators and mean accuracy score
    return imp, scr0.mean()

def auxFeatImpSFI(featNames, clf, trnsX, cont, scoring, cvGen):
    """
    Implementation of Single Feature Importance (SFI)
    Basically is to calculate all the cvScores of all the features
    :params featNames:
    :params clf:
    :params trnsX:
    :params cont:
    :params scoring:
    :params cvGen:
    
    :return imp:
    """
    # create an empty dataframe with column 'mean' and 'std'
    imp = pd.DataFrame(columns = ['mean', 'std'])
    for featName in featNames: # for each feature
        from cvFin import cvScore # import cvScore
        # cal the cvScores
        df0 = cvScore(clf,X = trnsX[[featName]], y = cont['bin'], sample_weight = cont['w'], scoring = scoring, cvGen = cvGen)
        # find the mean cvScores
        imp.loc[featName, 'mean'] = df0.mean()
        # find the std of cvScores
        imp.loc[featName, 'std'] = df0.std() * df0.shape[0]**-.5
    return imp

def get_eVec(dot, varThres):
    """
    compute eVec from dot prod matrix, reduce dimension
    :params dot:
    :params varThres:
    :return
        + eVal: eigen values
        + eVec: eigen vectors
    """
    #1) compute eVec and eVal from dot
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[ : :-1] # arguments for sorting eVal desc
    eVal, eVec = eVal[idx], eVec[ : , idx]
    #2) only positive eVals
    # eigen values are put into a pd.series, rank from most important to the least important
    eVal = pd.Series(eVal, index = ['PC_' + str(i + 1) for i in range(eVal.shape[0])])
    # eigen vectors are put into a pd.df, index = dot, columns = eVal
    eVec = pd.DataFrame(eVec, index = dot.index, columns = eVal.index)
    # ? in case there are additional columns, discard them all
    eVec = eVec.loc[:, eVal.index]
    #3) reduce dimension, form PCs
    # calculate and standardise the cumsum of the eval
    cumVar = eVal.cumsum() / eVal.sum()  
    # find the index of last cumsum that < varThres
    dim = cumVar.values.searchsorted(varThres)
    # [0: dim] are the eVal and eVec important
    eVal, eVec = eVal.iloc[: dim + 1], eVec.iloc[ : , : dim + 1]
    return eVal, eVec
#-----------------------------------------------------------------
def orthoFeats(dfX, varThres = .95):
    """
    Given a dataframe dfX of features, compute orthofeatures dfP
    :params dfX: pd.df, features
    :params varThres: float, threshold to select the significant Principal components

    :return
        dfP: pd.df, orthofeatures
    """
    # standardized features
    dfZ = dfX.sub(dfX.mean(), axis = 1).div(dfX.std(), axis = 1)
    # calculate the ZZ`(dot)
    dot = pd.DataFrame(np.dot(dfZ.T, dfZ), index = dfX.columns, columns = dfX.columns)
    # find the (significant) eVal and eVec
    eVal, eVec = get_eVec(dot, varThres)
    # get the orthofeatures
    dfP = np.dot(dfZ, eVec)
    return dfP

# COMPUTATION OF WEIGHTED KENDALL’S TAU BETWEEN FEATURE IMPORTANCE AND INVERSE PCA RANKING
# >>> import numpy as np
# >>> from scipy.stats import weightedtau
# >>> featImp=np.array([.55,.33,.07,.05]) # feature importance
# >>> pcRank=np.array([1,2,4,3]) # PCA rank
# >>> weightedtau(featImp,pcRank**-1.)[0]

def getTestData(n_features = 40, n_informative = 10, n_redundant = 10, n_samples = 10000):
    # generate a random dataset for a classification problem
    from sklearn.datasets import make_classification
    # trnsX = X, cont = y, informative features = `n_informative`, redundant features = `n_redundant` 
    trnsX, cont = make_classification(n_samples = n_samples, n_features = n_features, n_informative = n_informative, n_redundant = n_redundant, random_state = 0, shuffle = False)
    # n_samples days, freq = Business days, end by today
    df0 = pd.DatetimeIndex(periods = n_samples, freq = pd.tseries.offsets.BDay(), end = pd.datetime.today())
    # transform trnsX and cont into pd.df, cont.column.name = "bins"
    trnsX, cont = pd.DataFrame(trnsX, index = df0), pd.Series(cont, index = df0).to_frame('bin')
    # first n_informative are informative features, after that, first (n_redundant) are redundant features
    df0 = ['I_' + str(i) for i in range(n_informative)] + ['R_' + str(i) for i in range(n_redundant)]
    # the rest are noise features
    df0 += ['N_'+str(i) for i in range(n_features - len(df0))]
    # set trnsX.columns name
    trnsX.columns = df0
    # equal weight for each label
    cont['w'] = 1. / cont.shape[0]
    # vertical barrier is cont.index it self
    cont['t1'] = pd.Series(cont.index, index = cont.index)
    return trnsX, cont

def featImportance(trnsX, cont,
                   n_estimators = 1000, cv = 10, max_samples = 1., numThreads = 24, pctEmbargo = 0,
                   scoring = 'accuracy', method = 'SFI', minWLeaf = 0., **kargs):
    # feature importance from a random forest
    # bagged decision trees as default classifier
    n_jobs = (-1 if numThreads > 1 else 1) # run 1 thread with ht_helper in dirac1
    #1) prepare classifier,cv. max_features=1, to prevent masking
    clf = DecisionTreeClassifier(criterion = 'entropy', max_features = 1, class_weight = 'balanced', min_weight_fraction_leaf = minWLeaf)
    # A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
    # 'oob_score = True' use out-of-bag samples to estimate the generalization error.
    clf = BaggingClassifier(base_estimator = clf, n_estimators = n_estimators, max_features = 1., max_samples = max_samples, oob_score = True, n_jobs = n_jobs)
    # fit the data
    fit = clf.fit(X = trnsX, y = cont['bin'], sample_weight = cont['w'].values)
    # oob = Score of the training dataset obtained using an out-of-bag estimate.
    oob = fit.oob_score_
    if method == 'MDI': # MDI
        # MDI feature importance
        imp = featImpMDI(fit, featNames = trnsX.columns)
        # oos = CV
        oos = cvScore(clf, X = trnsX, y = cont['bin'], cv = cv, sample_weight = cont['w'], t1 = cont['t1'], pctEmbargo = pctEmbargo, scoring = scoring).mean()
    elif method == 'MDA': # MDA
        # MDA feature importance and oos
        imp, oos = featImpMDA(clf, X = trnsX, y = cont['bin'], cv = cv, sample_weight = cont['w'],t1 = cont['t1'], pctEmbargo = pctEmbargo, scoring = scoring)
    elif method == 'SFI': # SFI
        # CV
        cvGen = PurgedKFold(n_splits = cv, t1 = cont['t1'], pctEmbargo = pctEmbargo)
        # oos score
        oos = cvScore(clf, X = trnsX, y = cont['bin'], sample_weight = cont['w'], scoring = scoring, cvGen = cvGen).mean()
        clf.n_jobs = 1 # paralellize auxFeatImpSFI rather than clf
        # find the SFI imp
        imp = mpPandasObj(auxFeatImpSFI, ('featNames', trnsX.columns), numThreads, clf = clf, trnsX = trnsX, cont = cont, scoring = scoring, cvGen = cvGen)
    return imp, oob, oos


    
def testFunc(n_features = 40, n_informative = 10, n_redundant = 10,
         n_estimators = 1000, n_samples = 10000, cv = 10):
    # test the performance of the feat importance functions on artificial data
    # Nr noise features = n_features—n_informative—n_redundant
    # generate synthetic data
    trnsX, cont = getTestData(n_features, n_informative, n_redundant, n_samples)
    # arguments
    dict0 = {'minWLeaf': [0.], 'scoring': ['accuracy'], 'method': ['MDI','MDA', 'SFI'], 'max_samples': [1.]}    
    # split dict0 into 3 different jobs (by method)
    jobs =(dict(zip(dict0, i)) for i in product(*dict0.values()))
    out = [] # empty list
    # key arguments
    kargs = {'pathOut': './testFunc/', 'n_estimators': n_estimators, 'tag': 'testFunc', 'cv': cv}
    for job in jobs: # for each jobs
        # job params
        job['simNum'] = job['method'] + '_' + job['scoring'] + '_'+ '%.2f'%job['minWLeaf'] + '_' + str(job['max_samples'])
        print (job['simNum']) # print job params
        kargs.update(job) # update/add the elemets to the dictionary
        imp, oob, oos = featImportance(trnsX = trnsX, cont = cont, **kargs) #  find faeture importance using imp, oob, oos
        plotFeatImportance(imp = imp, oob = oob, oos = oos, **kargs) # plot the feature importance
        df0 = imp[['mean']] / imp['mean'].abs().sum() # normalised
        df0['type'] = [i[0] for i in df0.index] # 
        df0 = df0.groupby('type')['mean'].sum().to_dict() 
        df0.update({'oob': oob, 'oos': oos}) # update/add the elemets to the dictionary
        df0.update(job) # update/add the elemets to the dictionary
        out.append(df0) # append df0 to out
    out = pd.DataFrame(out).sort_values(['method', 'scoring', 'minWLeaf', 'max_samples']) # sort the df by
#     # only the followings are output
    out = out[['method', 'scoring', 'minWLeaf', 'max_samples', 'I', 'R', 'N', 'oob', 'oos']]
#    # out = out['method', 'scoring', 'minWLeaf', 'max_samples', 'oob', 'oos']
    out.to_csv(kargs['pathOut'] + 'stats.csv')
    return



def plotFeatImportance(pathOut, imp, oob, oos, method, tag = 0, simNum = 0, **kargs):
    # plot mean imp bars with std
    plt.figure(figsize = (10, imp.shape[0] / 5.))
    # sort imp['mean'] from low to high
    imp = imp.sort_values('mean', ascending = True)
    # plot horizontal bar
    ax = imp['mean'].plot(kind = 'barh', color = 'b', alpha = .25, xerr = imp['std'],error_kw = {'ecolor': 'r'})
    if method == 'MDI': # for MDI
        plt.xlim([0, imp.sum(axis = 1).max()])
        plt.axvline(1. / imp.shape[0], linewidth = 1, color = 'r', linestyle = 'dotted')
    ax.get_yaxis().set_visible(False) # disable y-axis
    for i, j in zip(ax.patches, imp.index): # label
        ax.text(i.get_width() / 2, i.get_y() + i.get_height() / 2, j, ha = 'center', va = 'center', color = 'black')
    plt.title('tag=' + tag + ' | simNum= '+ str(simNum) +' | oob=' + str(round(oob, 4)) + ' | oos=' + str(round(oos, 4)))
    plt.savefig(pathOut + 'featImportance_' + str(simNum) + '.png', dpi = 100)
    plt.clf()
    plt.close()
    return

