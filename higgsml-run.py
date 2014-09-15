#-------------------------------------------------------------------------------
# Name:        higgsml-run.py
# Purpose:     Run the Higgs classifier
#
# Author:      Alexander Lavin
#
# Created:     15/09/2014
# Copyright:   (c) Alexander Lavin 2014
#              alexanderlavin.com
#-------------------------------------------------------------------------------

def main(gbc):

    import math
    import pandas as pd
    import numpy as np

    # Run model on test data and export results csv
    print 'Loading and running testing data, writing to csv'
    data = pd.read_csv("test.csv")
    X_test = data.values[:, 1:]

    ids = data.EventId
    d = gbc.predict_proba(X_test)[:, 1]

    r = np.argsort(d) + 1 # argsort(d) returns the indices that would sort the array d
    p = np.empty(len(X_test), dtype=np.object)
    p[d > pcut] = 's'
    p[d <= pcut] = 'b'

    df = pd.DataFrame({"EventId": ids, "RankOrder": r, "Class": p})
    df.to_csv("predictions.csv", index=False, cols=["EventId", "RankOrder", "Class"])

    return []

if __name__ == '__main__':
    main(model)
    pass
