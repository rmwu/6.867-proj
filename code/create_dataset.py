import numpy as np
import math
from load_data import *

# june = load_month("06", "15",
#                   max_rows = 4000)

##########################################
#   DATA RELOADING
##########################################

def reload_pretty_data(month, year,
                      classify=False):
    """
    reload_pretty_data loads already prettified data and splits
    the data into X (all columns but last) and y (last column)
    such that X[i] is the feature vector for flight i, with lateness
    in y[i]
    
    classify       True, loads up discrete data (-1,1) labels for y,
                   False, loads up continuous labels for y (lateness)
    """
    if classify:
        filename = "data/{}_{}_classify.csv".format(year, month)
    else:
        filename = "data/{}_{}_regression.csv".format(year, month)
    
    data = np.loadtxt(
        fname = filename,
        delimiter = ",")
    
    return (data[:,:-1],data[:,-1])

##########################################
#   DATASET GENERATION
##########################################

def pretty_data(data, maps = None, classify=False):
    if maps is None:
        maps = [lambda x:x]*2
        
    # stick your functions here
    flight_dates = [0 for x in data["FL_DATE"]] # maps[0](data["FL_DATE"])
    departure_times = data["CRS_DEP_TIME"].astype(str)
    # hella jank conversion...
    # departure_times = [int(x[1:-1])/2400 for x in departure_times]
    departure_times = [int(x[1:-1]) for x in departure_times]
                    
    # maps[1](data["CRS_DEP_TIME"])
    
    airlines = convert_to_onehot(data["AIRLINE_ID"])[0]
    origins = convert_to_onehot(data["ORIGIN_AIRPORT_ID"])[0]
    destinations = convert_to_onehot(data["DEST_AIRPORT_ID"])[0]
    
    delays = data["DEP_DELAY"]
    
    X = []
    n = len(data)
    
    for i in range(n):
        # only bother if delay available
        if math.isnan(delays[i]):
            continue
            
        vector = [flight_dates[i], departure_times[i]]
        
        onehots = np.append(airlines[i],origins[i])
        onehots = np.append(onehots,destinations[i])
        vector.extend(onehots.tolist())
        
        # if we want to classify, delays become binary {-1,1}
        if classify:
            delay_binary = -1 if delays[i] >= 0 else 1
            vector.append(delay_binary)
        else:
            vector.append(delays[i])
        
        X.append(vector)
    
    X = np.array(X)#.astype("float_")
    
    # different filename for classification
    if classify:
        filename = "data/15_06_classify.csv"
    else:
        filename = "data/15_06_regression.csv"
    np.savetxt(filename, X, delimiter=",",fmt='%i') # 
    return X

##########################################
#   DATA TRANSFORMATIONS
##########################################

def date_to_cyclic(datestring):
    """
    date_to_cyclic converts a date, formatted as a string, into
    a numpy feature vector (???)
    
    datestring    string formatted as however the raw data is
    
    returns       numpy vector, however you want the cyclic date to be
                  represented in the final flight feature vector
    """
    pass

def time_to_cyclic(time_into_day):
    """
    time_into_day  integer between 0 and 2400, such that the int hhmm
                   represents the time at hh:mm
                   
    returns       numpy vector, however you want the time to be
                  represented in the final flight feature vector
                  (does not have to be cyclic; just something other than
                  the int, which isn't even smooth)
    """
    pass