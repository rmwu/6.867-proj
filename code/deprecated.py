"""
graveyard for discarded, neglected code
"""

def create_dataset(data, 
                   features, label,
                   makeFeatures = None,
                   makeLabel = lambda x:x):
    """
    create_dataset returns a tuple pair of X and y, containing the
    features, whose "header" is identified in features, and labels,
    whose "header" is identified in labels
    
    features    list of indices, corresponding to columns of headers, which
                will identify the features concatenated together
    label       index into headers, which will identify column of data
                used to generate labels. If None, then the identity will be
                be used.
    makeLabel   function to map label value in data, to the label used in
                the actual ML. If not provided, then the identity will be used.
    """
    if makeFeatures is None:
        makeFeatures = [lambda x:x] * len(features)
    
    X = [featureMap(data[headers[feature]])
                for (feature, featureMap) 
                in list(zip(features, makeFeatures))]
    
    # transpose X such that X[i] is sample i
    X = np.array(X).T
    
    # element-wise apply labels
    y = np.array([makeLabel(x) for x in data[headers[label]]])
    
    return (X, y)


"""
X_june, y_june = create_dataset(june,
    [
    headers.index("CRS_DEP_TIME"),
    headers.index("FL_DATE"),
    headers.index("AIRLINE_ID"),
    headers.index("ORIGIN_AIRPORT_ID"),
    headers.index("DEST_AIRPORT_ID"),
    ],
    headers.index("DEP_DELAY"),
    # convert nan to 0, positive to 1, negative to 0
    makeLabel = lambda x : 0 if math.isnan(x) else min(max(x,0),1))
"""