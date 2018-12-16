import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,accuracy_score
from cvFin import PurgedKFold, cvScore
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

class MyPipeline(Pipeline):
    """
    Inherit all methods from sklearn's `Pipeline`
    Overwrite the inherited `fit` method with a new one that handles the argument `sample weight`
    After which it redirects to the parent class
    """
    def fit(self, X, y, sample_weight = None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)

def clfHyperFit(feat, lbl, t1, pipe_clf, param_grid,
                cv = 3, bagging = [0, None, 1.], n_jobs = -1, pctEmbargo = 0, **fit_params):
    """
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params: 
    :return gs:
    """
    scoring = 'f1' # f1 for meta-labeling
    # if set(lbl.values) == {0, 1}: # if label values are 0 or 1
    #     scoring = 'f1' # f1 for meta-labeling
    # else:
    #     scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    # perform grid search
    gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid, 
                        scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    # best estimator and the best parameter
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1] > 0: # max_samples > 0
        gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps),
                                n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                max_features = float(bagging[2]), n_jobs = n_jobs)
        gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs

def clfHyperFitRand(feat, lbl, t1, pipe_clf, param_grid, cv=3, bagging=[0,None,1.], rndSearchIter=0, n_jobs=-1, pctEmbargo=0, **fit_params):
    """
    Randimised Search with Purged K-fold CV
    Grid Search with purged K-Fold Cross Validation
    :params feat: features
    :params lbl: labels
    :params t1: vertical barriers, used for PurgedKFold
    :params pipe_clf: classification pipeline
    :params param_grid: parameter grid
    :params cv: int, cross validation fold
    :params bagging: bagging parameter?
    :params rndSearchIter
    :params n_jobs: CPUs
    :params pctEmbargo: float, % of embargo
    :params **fit_params: 
    :return gs:
    """
    if set(lbl.values) == {0,1}:# if label values are 0 or 1
        scoring = 'f1' # f1 for meta-labeling
    else:
        scoring = 'neg_log_loss' # symmetric towards all cases
    #1) hyperparameter search, on train data
    # prepare the training sets and the validation sets for CV (find their indices)
    inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged
    if rndSearchIter == 0: # randomised grid search
        gs = GridSearchCV(estimator = pipe_clf, param_grid = param_grid,
                            scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False)
    else: # normal grid search
        gs = RandomizedSearchCV(estimator = pipe_clf, param_distributions = param_grid, 
                                scoring = scoring, cv = inner_cv, n_jobs = n_jobs, 
                                iid = False, n_iter = rndSearchIter)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline
    #2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps),
                                n_estimators = int(bagging[0]), max_samples = float(bagging[1]),
                                max_features = float(bagging[2]), n_jobs = n_jobs)
        gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
        gs = Pipeline([('bag', gs)])
    return gs


from scipy.stats import rv_continuous,kstest
#———————————————————————————————————————
class logUniform_gen(rv_continuous):
# random numbers log-uniformly distributed between 1 and e
    def _cdf(self,x):
        return np.log(x/self.a)/np.log(self.b/self.a)
def logUniform(a=1,b=np.exp(1)):return logUniform_gen(a=a,b=b,name='logUniform')
#———————————————————————————————————————
# a,b,size=1E-3,1E3,10000
# vals=logUniform(a=a,b=b).rvs(size=size)
# print kstest(rvs=np.log(vals),cdf='uniform',args=(np.log(a),np.log(b/a)),N=size)
# print pd.Series(vals).describe()
# plt.subplot(121)
# pd.Series(np.log(vals)).hist()
# plt.subplot(122)
# pd.Series(vals).hist()
# plt.show()