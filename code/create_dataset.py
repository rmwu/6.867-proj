import numpy as np
import math
from load_data import *

# june = load_month("06", "15",
#                   max_rows = 4000)

def pretty_data(data, maps = None):
    if maps is None:
        maps = [lambda x:x]*2
        
    # stick your functions here
    flight_dates = [0 for x in data["FL_DATE"]] # maps[0](data["FL_DATE"])
    departure_times = data["CRS_DEP_TIME"].astype(str)
    # hella jank conversion...
    departure_times = [int(x[1:-1])/2400 for x in departure_times]
                    
    # maps[1](data["CRS_DEP_TIME"])
    
    airlines = convert_to_onehot(data["AIRLINE_ID"])[0]
    origins = convert_to_onehot(data["ORIGIN_AIRPORT_ID"])[0]
    destinations = convert_to_onehot(data["DEST_AIRPORT_ID"])[0]
    
    delays = data["DEP_DELAY"]
    
    X = []
    n = len(data)
    
    for i in range(n):
        vector = [flight_dates[i], departure_times[i]]
        
        onehots = np.append(airlines[i],origins[i])
        onehots = np.append(onehots,destinations[i])
        vector.extend(onehots.tolist())
        
        vector.append(delays[i])
        
        X.append(vector)
    
    X = np.array(X).astype("float_")
    np.savetxt("data/15-06-clean.csv", X, delimiter=",") # ,fmt='%i'
    return X