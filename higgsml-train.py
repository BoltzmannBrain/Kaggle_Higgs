#-------------------------------------------------------------------------------
# Name:        higgsml-train.py
# Purpose:     Train Higgs classifier
#
# Author:      Alexander Lavin
#
# Created:     15/09/2014
# Copyright:   (c) Alexander Lavin 2014
#              alexanderlavin.com
#-------------------------------------------------------------------------------

def main():
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.svm import SVC
    import math
    import pandas as pd
    from sklearn import preprocessing

    # Load training data
    print 'Loading training data.'
    data_train = np.loadtxt( 'training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )

    # Pick a random seed for reproducible results
    np.random.seed(42)
    # Random number for training/validation splitting
    r =np.random.rand(data_train.shape[0])

    # Put Y(truth), X(data), W(weight), and I(index) into their own arrays
    print 'Assigning data to numpy arrays.'
    # The 90% of samples where r<0.9 are training:
    Y_train = data_train[:,32][r<0.9]
    X_train = data_train[:,1:31][r<0.9]
    W_train = data_train[:,31][r<0.9]
    # The other 10% are validation:
    Y_valid = data_train[:,32][r>=0.9]
    X_valid = data_train[:,1:31][r>=0.9]
    W_valid = data_train[:,31][r>=0.9]

    # Train initial estimator
    class Adopter(object):
        """A class that adapts a ClassifierMixin's interface to what GBRT expects. """
        def __init__(self, obj):
           self.obj = obj
        def predict(self, X):
            # output needs to be 2D...
            return self.obj.predict_proba(X)[:, 1:]
        def fit(self, X, y):
            self.obj.fit(X, y)
            return self
    print 'Training base estimator...'
    gbc_init = Adopter(SGDClassifier(loss='log',n_iter=5))

    # Train the GradientBoostingClassifier
    print 'Training GBC classifier...'
    gbc = GBC(n_estimators=13000, max_depth=10,min_samples_leaf=100,max_features=10,learning_rate=0.0005,verbose=1,init=gbc_init) # include base estimator w/ init
    gbc.fit(X_train,Y_train)

    # Get the probaility output from the trained method, using the 10% for testing
    prob_predict_train = gbc.predict_proba(X_train)[:,1]
    prob_predict_valid = gbc.predict_proba(X_valid)[:,1]

    # Choose top 15% as signal gives a good AMS score
    pcut = np.percentile(prob_predict_train,85)

    # These are the final signal and background predictions
    Yhat_train = prob_predict_train > pcut # set of events in the training set that are classified as signal events
    Yhat_valid = prob_predict_valid > pcut # set of events in the validation set that are classified as signal events

    # To calculate the AMS data, first get the true positives and true negatives
    # Scale the weights according to the r cutoff.
    TruePositive_train = W_train*(Y_train==1.0)*(1.0/0.9)
    TrueNegative_train = W_train*(Y_train==0.0)*(1.0/0.9)
    TruePositive_valid = W_valid*(Y_valid==1.0)*(1.0/0.1)
    TrueNegative_valid = W_valid*(Y_valid==0.0)*(1.0/0.1)

    # s and b for the training
    s_train = sum ( TruePositive_train*(Yhat_train==1.0) )
    b_train = sum ( TrueNegative_train*(Yhat_train==1.0) )
    s_valid = sum ( TruePositive_valid*(Yhat_valid==1.0) )
    b_valid = sum ( TrueNegative_valid*(Yhat_valid==1.0) )

    # Calculate the AMS scores
    print 'Calculating AMS score for a probability cutoff pcut=',pcut
    def AMSScore(s,b): return  math.sqrt (2.*( (s + b + 10.)*math.log(1.+s/(b+10.))-s))
    print '   - AMS based on 90% training   sample:',AMSScore(s_train,b_train)
    print '   - AMS based on 10% validation sample:',AMSScore(s_valid,b_valid)

    return gbc

if __name__ == '__main__':
    model = main()
    pass
