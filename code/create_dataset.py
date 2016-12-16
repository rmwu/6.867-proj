import numpy as np
import math

import time
import calendar

from load_data import *

# june = load_month("06", "15",
#                   max_rows = 4000)

##########################################
#   DATA RELOADING
##########################################

def reload_pretty_data(month, year,
                      classify=False,
                      shuffle=True):
    """
    reload_pretty_data loads already prettified data and splits
    the data into X (all columns but last) and y (last column)
    such that X[i] is the feature vector for flight i, with lateness
    in y[i]
    
    classify       True, loads up discrete data (-1,1) labels for y,
                   False, loads up continuous labels for y (lateness)
    shuffle        True to shuffle
    """
    if classify:
        filename = "data/{}_{}_classify.csv".format(year, month)
    else:
        filename = "data/{}_{}_regression.csv".format(year, month)
    
    data = np.loadtxt(
        fname = filename,
        delimiter = ",")
    
    dims = data.shape
    if shuffle:
        np.random.shuffle(data)
        assert data.shape == dims
    
    return (data[:,:-1],data[:,-1])

##########################################
#   DATASET GENERATION
##########################################

def pretty_data(data, maps = None, classify=False):
    if maps is None:
        maps = [lambda x:x]*2
    
    mappings = load_latlong()
        
    # stick your functions here
    flight_dates = list(map(date_to_cyclic, map(str, data["FL_DATE"].astype(str))))
    # [date_to_cyclic(str(date)) for date in data["FL_DATE"].astype(str)]
    departure_times = list(map(time_to_cyclic, map(lambda x : str(x)[1:-1], data["CRS_DEP_TIME"].astype(str))))
        
    # hella jank conversion...lol    

    airlines = convert_to_onehot(data["AIRLINE_ID"])[0]
    # origins = convert_to_onehot(data["ORIGIN_AIRPORT_ID"])[0]
    # destinations = convert_to_onehot(data["DEST_AIRPORT_ID"])[0]
    
    origin_strs = data["ORIGIN_AIRPORT_ID"].astype(str).tolist()
    dest_strs = data["DEST_AIRPORT_ID"].astype(str).tolist()
    
    latlongs_orig = list(map(lambda x : airport_to_latlong(x, mappings),
                             origin_strs))
    latlongs_dest = list(map(lambda x : airport_to_latlong(x, mappings),
                             dest_strs))
    
    delays = data["DEP_DELAY"]
    
    X = []
    n = len(data)
    
    for i in range(n):
        # only bother if delay available
        if math.isnan(delays[i]):
            continue
            
        vector = np.append(flight_dates[i], departure_times[i]).tolist()
        
        # onehots = np.append(airlines[i],origins[i])
        # onehots = np.append(onehots,destinations[i])
        # vector.extend(onehots.tolist())
        
        vector.extend(latlongs_orig[i])
        vector.extend(latlongs_dest[i])
        
        displacement = [vector[-1]-vector[-3],
                        vector[-2]-vector[-4]]
        # vector.extend(displacement)
        
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
        
    np.savetxt(filename, X, delimiter=",") # ,fmt='%i'
    return X

def date_to_cyclic(date_str):
    """
    date_to_cyclic converts a date, formatted as a string, into
    a numpy feature vector (???)
    
    date_str    string formatted as however the raw data is
    
    returns       numpy vector, however you want the cyclic date to be
                  represented in the final flight feature vector
    """
    assert type(date_str) is str

    time_struct = time.strptime(date_str, "%Y-%m-%d")
    days_in_year = 366 if calendar.isleap(time_struct.tm_year) else 365
    angular_year_frac = 2 * math.pi * time_struct.tm_yday / days_in_year

    days_in_month = calendar.monthrange(time_struct.tm_year, time_str.tm_month)
    angular_month_frac = 2 * math.pi * time_struct.tm_mday / days_in_month

    angular_week_frac = 2 * math.pi * time_struct.tm_wday / 7

    return np.array([math.cos(angular_year_frac), math.sin(angular_year_frac),
        math.cos(angular_month_frac), math.sin(angular_month_frac),
        math.cos(angular_week_frac), math.sin(angular_week_frac)])

def time_to_cyclic(time_str):
    """
    time_str    a string "HHMM" representing the 24-hour time.
                   
    returns       numpy vector, however you want the time to be
                  represented in the final flight feature vector
                  (does not have to be cyclic; just something other than
                  the int, which isn't even smooth)
    """
    assert type(time_str) is str

    time_struct = time.strptime(time_str, "%H%M")
    minutes_elapsed = 60 * time_struct.tm_hour + time_struct.tm_min
    minutes_in_day = 60 * 24
    angular_day_frac = 2 * math.pi * minutes_elapsed / minutes_in_day
    return np.array([math.cos(angular_day_frac), math.sin(angular_day_frac)])



def airport_to_latlong(airportID, mappings):
    """
    airportID    int US Dept of Trans id for airport, unique identifier
    mappings     output from load_latlong
    
    returns      tuple (lat, long)
    """
    if airportID in mappings:
        return map(float, mappings[airportID])
    return [0,0]
    
def load_latlong():
    """
    loads up latlong csv and returns a np array
    """
    filename = "data/airports.csv"
    
    data = np.genfromtxt(
        fname = filename,
        delimiter = ",", # csv files
        dtype = None, # figure out string vs. numeric
        names = True  # read names from first row
    )
    
    # AIRPORT_ID LATITUDE LONGITUDE
    airports = data["AIRPORT_ID"].astype(str)
    lats = data["LATITUDE"].astype(str)
    longs = data["LONGITUDE"].astype(str)
    
    assert airports.shape == lats.shape
    assert lats.shape == longs.shape
    
    mappings = {}
    for (air, lat, longg) in list(zip(airports.tolist(), lats, longs)):
        mappings[str(air)] = [lat, longg]
        
    assert type(mappings) is dict
    return mappings