from scipy.misc import comb
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#------------------------------------------------------------------------------------------------
# ACCURACY OF THE BAGGING CLASSIFIER
N,p,k=100,1./3,3.
p_=0
for i in range(0,int(N/k)+1):
    p_+=comb(N,i)*p**i*(1-p)**(N-i)
print (p,1-p_)
#-------------------------------------------------------------------------------------------------
# Three ways of setting up an RF

# 'max_features' = lower value -> force discrepancy between trees
# 'min_weight_fraction_leave' = large value (e.g. 5%) -> out-of-bag accuracy converges to out=of-sample (k-fold) accuracy
# Use `BaggingClassifier`  on `DecisionTreeClassifier` or `RandomForestClassfier` where: 
#   `max_samples` is set to the `avgU` (average uniqueness) between samples
# Modify the RF class to replace standard bootstrapping with sequential bootstrapping


# clf0=RandomForestClassifier(n_estimators=1000,class_weight='balanced_subsample',criterion='entropy')
# clf1=DecisionTreeClassifier(criterion='entropy',max_features='auto',class_weight='balanced')
# clf1=BaggingClassifier(base_estimator=clf1,n_estimators=1000,max_samples=avgU)
# clf2=RandomForestClassifier(n_estimators=1,criterion='entropy',bootstrap=False,class_weight='balanced_subsample')
# clf2=BaggingClassifier(base_estimator=clf2,n_estimators=1000,max_samples=avgU,max_features=1.)