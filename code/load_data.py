import numpy as np
import csv

# global constants
variables = {
    "NUM_TRAIN" : 20000,
    "NUM_VALIDATE": 5000,
    "NUM_TEST" : 5000
}
headers = ["YEAR","MONTH","FL_DATE",
"AIRLINE_ID","CARRIER",
"ORIGIN_AIRPORT_ID","ORIGIN_CITY_MARKET_ID","ORIGIN_WAC",
"DEST_AIRPORT_ID","DEST_CITY_MARKET_ID",
"CRS_DEP_TIME","DEP_TIME","DEP_DELAY",
"CRS_ARR_TIME","ARR_TIME","ARR_DELAY",
"CANCELLED","CANCELLATION_CODE",
"CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY",
"SECURITY_DELAY","LATE_AIRCRAFT_DELAY"]

# data input
def load_month(month, year, max_rows = variables["NUM_TRAIN"]):
    """
    load_month loads and preprocesses the airline data
    associated with the given month and year, which must
    be passed in as STRINGS of MM YY format.
    
    max_rows    maximum number of rows to load

    For example, the following code loads 10 rows from June '15
        june = load_month("06", "15", max_rows = 10)
    """
    filename = "data/{}_{}.csv".format(year, month)
    
    data = np.genfromtxt(
        fname = filename,
        delimiter = ",", # csv files
        dtype = None, # figure out string vs. numeric
        names = True, # read names from first row
        max_rows = max_rows)
    
    return data

def convert_to_onehot(categories):
    """
    convert_to_onehot takes in a list of discrete values,
    each in a separate category, and returns
    1) a numpy array of one hot vectors, in the same order
       as the categories,
    2) a dict, mapping original category to one hot vector

    For example, the following code loads the arrival time
    as one hot vectors
        arrival_times = convert_to_onehot(june["ARR_TIME"])
    """
    unique_types = np.unique(categories) # set of categories
    num_types = unique_types.shape[0] # total elements in set
    identity = np.identity(num_types) # extract one hot vectors
    type_mapping = {} # maps original category to one hot
    
    # generate mapping from type to one hot vector
    for i in range(num_types):
        type_mapping[unique_types[i]] = identity[i]
    
    # use our new dict to map one hot vectors
    converted_types = []
    for category in categories:
        converted_types.append(type_mapping[category])
        
    return (np.array(converted_types), type_mapping)