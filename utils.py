import pandas as pd
import numpy as np
import random
import torch
import gurobipy
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle as pkl
import utils
import algs
from sodapy import Socrata
from datetime import datetime, timedelta

from math import sin, cos, asin, sqrt, pi

# Returns distance between two points in METERS
def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742000 * asin(sqrt(a)) #2*R*asin...

# Generates a uniform random point in a circle of radius r METERS around a point with longitude/latitude coordinates 
# x0, y0
# def rand_point(lat, lon, r):
def rand_point(y0,x0,distance):
    r = distance/ 111300
    u = np.random.uniform(0,1)
    v = np.random.uniform(0,1)
    w = r * np.sqrt(u)
    t = 2 * np.pi * v
    x = w * np.cos(t)
    x1 = x / np.cos(y0)
    y = w * np.sin(t)
    return (y0+y, x0 + x1)


def to_str(datetime):
    s = str(datetime)
    return "T".join(s.split())

def flatten(lst):
    return [element for sublst in lst for element in sublst]

'''
Fetches data from Chicago and returns dataframes.
Input:
    limit (default 10000): maximum number of rows in each dataframe
Output:
    D1_df: DataFrame where each row corresponds to a first-stage demand vertex (ride request)
    D2_df_list: List of DataFrames, where each dataframe in the list corresponds to a possible second-stage scenario.
    S_df: DataFrame where each row is a supply vertex (driver).
These are generated as follows:
    1. Sample a random day between May 1, 2022 and May 15, 2022
    2. Sample a random time between 10am and 4pm (15 min ticks)
    3. D1 is the set of all rides that STARTED at that day/time.
    4. S is the set of all rides that FINISHED at time t-15 on that day.
    5. For each day between May 1, 2022 and May 15, 2022, generate a possible
        second-stage demand D2 by taking all rides that STARTED at time t+15
          on that day.
'''
def get_data(limit=10000):
    client = Socrata("data.cityofchicago.org", None)

    DATASET = "2tdj-ffvb"

    year = 2022
    month = 4
    
    day_lo = 1
    day_hi = 31
    
    day = np.random.randint(day_lo, day_hi)
    day_prime = np.random.randint(day_lo, day_hi)
    
    hour = np.random.randint(10, 16)
    minute = np.random.randint(0, 4) * 15
    date = datetime(year, month, day, hour, minute)
    date_prime = datetime(year, month, day_prime, hour, minute)
    date_prev = date - timedelta(minutes=15)
    
    # Get all possible second stage demands
    D2_df_list = []

    loc_filter_pickup = "pickup_centroid_latitude between 41.8 and 42 and pickup_centroid_longitude between -87.7 and -87.6"
    loc_filter_dropoff = "dropoff_centroid_latitude between 41.8 and 42 and dropoff_centroid_longitude between -87.7 and -87.6"
    
    for d in range(day_lo, day_hi):
        date_prime = datetime(year, month, d, hour, minute) + timedelta(minutes=15)
        data = client.get(DATASET,
                       where="trip_start_timestamp = '{date}' and {loc_filter}".format(date=to_str(date_prime), loc_filter=loc_filter_pickup),
                       limit=limit)
        df = pd.DataFrame.from_records(data)
        df = df[df[['pickup_centroid_location']].notnull().all(1)]
        D2_df_list.append(df)
    
    D1_data = client.get(DATASET, 
                     where="trip_start_timestamp = '{date}' and {loc_filter}".format(date=to_str(date), loc_filter=loc_filter_pickup), 
                     limit=limit)

    S_data = client.get(DATASET,
                     where="trip_end_timestamp = '{date}' and {loc_filter}".format(date=to_str(date_prev), loc_filter=loc_filter_dropoff), 
                     limit=limit)
    
    D1_df = pd.DataFrame.from_records(D1_data)
    S_df = pd.DataFrame.from_records(S_data)
    
    # Do some filtering
    D1_df = D1_df[D1_df[['pickup_centroid_location']].notnull().all(1)]
    S_df = S_df[S_df[['dropoff_centroid_location']].notnull().all(1)]

    return D1_df, D2_df_list, S_df


"""
Returns a bipartite graph given ride request data.
The input dataframes have the following fields:
    trip_id
    trip_start_timestamp
    trip_end_timestamp
    trip_seconds
    trip_miles
    dropoff_community_area
    fare
    tip
    additional_charges
    trip_total
    shared_trip_authorized
    trips_pooled
    dropoff_centroid_latitude
    dropoff_centroid_longitude
    dropoff_centroid_location
    pickup_community_area
    pickup_centroid_latitude
    pickup_centroid_longitude
    pickup_centroid_location
    pickup_census_tract
    dropoff_census_tract
Arguments:
    D1_df (dataframe): The first-stage demand
    D2_df (dataframe): The second-stage demand
    S_df (dataframe): The supply
Output:
    edges1 (list of edges): The first-stage graph
    edges2 (list of edges): The second-stage graph
    edges2_list (list of list of edges): The list of possible second-stage graphs (for generating advice)
    weights: The weights of the offline vertices
"""
def create_graph(D1_df, D2_df_list, S_df, thresh=1.0, radius=1000):
    edges1 = []
    edges2_list = []
    
    D1_pickups = [p['coordinates'] for p in D1_df['pickup_centroid_location']]
    S_dropoffs = [p['coordinates'] for p in S_df['dropoff_centroid_location']]
    
    D1_pickups = list(map(lambda pt: rand_point(pt[0], pt[1], radius), D1_pickups))
    S_dropoffs = list(map(lambda pt: rand_point(pt[0], pt[1], radius), S_dropoffs))
    
    D2_pickups_list = []
    
    for day in range(len(D2_df_list)):
        D2_df = D2_df_list[day]
        D2_pickups = [p['coordinates'] for p in D2_df['pickup_centroid_location']]
        D2_pickups = list(map(lambda pt: rand_point(pt[0], pt[1], radius), D2_pickups))
        edges2 = []
                
        # Each demand can match to two closest neighbors
        
        for j in range(len(S_dropoffs)):
            for i in range(len(D2_pickups)):
                pickup_coord = D2_pickups[i]
                dropoff_coord = S_dropoffs[j]
                if distance(pickup_coord[0], pickup_coord[1], dropoff_coord[0], dropoff_coord[1]) < thresh:
                    edges2.append((i,j))
        edges2_list.append(edges2)
        
        D2_pickups_list.append(D2_pickups)
    
    for j in range(len(S_dropoffs)):
        for i in range(len(D1_pickups)):
            pickup_coord = D1_pickups[i]
            dropoff_coord = S_dropoffs[j]
            if distance(pickup_coord[0], pickup_coord[1], dropoff_coord[0], dropoff_coord[1]) < thresh:
                edges1.append((i,j))
    
    S = set([e[1] for e in edges1] + [e[1] for e in flatten(edges2_list)])
    
    weights = {j: 1 for j in S}
    
    idx = np.random.randint(0,len(edges2_list))
    
    edges2 = edges2_list[idx]
    
    coordinate_info = {'D1': D1_pickups, 
                       'D2': D2_pickups_list[idx],
                       'S': S_dropoffs}
    
    return edges1, edges2, edges2_list, weights, coordinate_info

'''
Corrupts the advice.
Input:
    edges1 (list of edges): Edges in the first-stage graph.
    sugg_matching (list of edges): Edges in the suggested matching.
    p (default 0.0): Probability to corrupt an edge in the suggested matching.
Output:
    A "corrupted" suggested matching, obtained by taking each edge (i,j) 
      in the original suggested matching and with probability p, replacing 
        it with a random edge (i, j') where j' is a random free vertex.
'''
def corrupt_advice(edges1, sugg_matching, p=0.0):
    D = set([e[0] for e in edges1])
    S = set([e[1] for e in edges1])
    nbrs_D = {i: [] for i in D}
    nbrs_S = {j: [] for j in S}
    corrupted_matching = []
    matched_already = [e[1] for e in sugg_matching]
    for i,j in edges1:
        nbrs_D[i].append(j)
        nbrs_S[j].append(i)
    for i,j in sugg_matching:
        u = np.random.uniform(0,1)
        if u < p:
            # corrupt the edge
            free_nbrs = [j for j in nbrs_D[i] if j not in matched_already] 
            
            if len(free_nbrs) == 0:
                # Just delete edge (i,j) from the suggested matching
                matched_already.remove(j)
            else:
                # Otherwise randomly choose available j' and add (i, j')
                # to the suggested matching
                idx = np.random.randint(len(free_nbrs))
                j_prime = free_nbrs[idx]
                corrupted_matching.append((i,j_prime))
                matched_already.remove(j)
                assert(j_prime not in matched_already)
                matched_already.append(j_prime)
        else:
            corrupted_matching.append((i,j))
            
    return corrupted_matching
'''
Plot the latitude/longitude coordinates of the demand/supply locations.
Input:
    D1_coords: List of coordinates of the riders in D1
    D2_coords: List of coordinates of the riders in D2
    S_coords: List of coordinates of the drivers in S
Output:
    Prints a scatter plot of the locations of D1, D2, and S. 
'''
def visualize(D1_coords, D2_coords, S_coords, savefig=False):
    plt.scatter([p[0] for p in D1_coords], [p[1] for p in D1_coords], label='D1', alpha=0.6)
    plt.scatter([p[0] for p in D2_coords], [p[1] for p in D2_coords], label='D2', alpha=0.6)
    plt.scatter([p[0] for p in S_coords], [p[1] for p in S_coords], label='S', alpha=0.6)
    plt.legend()
    plt.xlabel('Latitude')
    plt.ylabel("Longitude")
    plt.title("Coordinates of Riders (D1/D2) and Drivers (S)")
    if savefig:
        plt.savefig('{0}.png'.format("coords"), dpi=1000)
    plt.show()
    
'''
Input:
    raw_data: list of elements where each element looks like
            R     p
         {('0', '0.0'): 80.99999900946094,
          ('0.25', '0.0'): 81.0000665370493,
          ('0.5', '0.0'): 80.99953257238232,
          ('0.75', '0.0'): 78.28258212000853,
          ('0', '0.5'): 79.0000491358058,
          ('0.25', '0.5'): 79.00012011376225,
          ('0.5', '0.5'): 78.99969863616542,
          ('0.75', '0.5'): 78.19876344279089,
          ('0', '1'): 76.56532870672632,
          ('0.25', '1.0'): 76.90121378419238,
          ('0.5', '1.0'): 77.08005952927753,
          ('0.75', '1.0'): 78.11128417607173,
          'opt': 81.0,
          'advice': 80.99998762027245,
          'greedy': 78.0}
         
Output:
    A table of average competitive ratios
    
Basically I have p, r pairs
-- every Monte carlo replication
     -> Run opt, greedy, advice, all (p,r) pairs
-- output
     -> avg comp ratio of greedy, advice, and (p,r) for all p,r
'''    
def crs(raw_data):
    total_greedy = 0
    total_opt = 0
    total_advice = 0
    total_ours = {}
    Rp_list = [Rp for Rp in raw_data[0].keys() if Rp not in ['opt', 'advice', 'greedy']]
#     R_list = [R for R in raw_data[0].keys() if R not in ['opt', 'advice', 'greedy']]
    for d in raw_data:
        opt = d['opt']
        total_greedy += d['greedy'] / opt
        total_advice += d['advice'] / opt
        for Rp in Rp_list:
            if Rp not in total_ours:
                total_ours[Rp] = d[Rp]/opt
            else:
                total_ours[Rp] += d[Rp]/opt
    num_trials = len(raw_data)
    for Rp in total_ours:
        total_ours[Rp] = total_ours[Rp]/num_trials
    
    return total_ours, total_advice/num_trials, total_greedy/num_trials
    
    
'''
Given raw competitive ratio data for all the Monte Carlo replications,
generate a plot. 
Input:
    crs_data [dict, float, float]: List where
        - First entry is a dictionary where each key is an (R, p) pair and the value is the average CR
        - Second entry is the average CR of uncorrupted advice
        - Third entry is the average CR of greedy
Output:
    A plot with p on the x-axis and CR on the y-axis.
    A line for each value of R.
    A horizontal line for average CR of uncorrupted advice.
    A horizontal line for average CR of greedy. 
'''    
def plot_crs(crs_data, title=None, savefig=False):
    Rp_dict = crs_data[0]
    advice_cr = crs_data[1]
    greedy_cr = crs_data[2]
    robust_cr = crs_data[3]
    Rp_keys = Rp_dict.keys()
    p_range = sorted(list(set([rp[1] for rp in Rp_keys])))
    R_range = sorted(list(set([rp[0] for rp in Rp_keys])))
    
    alg_crs = {}
    for R in R_range:
        crs = []
        for p in p_range:
            crs.append(Rp_dict[(R,p)])
        alg_crs[R] = crs
    
    p_range = list(map(float, p_range))
    plt.plot(p_range, [1-greedy_cr]*len(p_range), linestyle='dotted', label='greedy', marker='.')
    plt.plot(p_range, [1-robust_cr]*len(p_range), linestyle='dotted', label='robust', marker='.')
    for R in reversed(R_range):
        if R != '0.0':
            plt.plot(p_range, 1-np.array(alg_crs[R]), label='R = ' + R, marker='.', alpha=0.8)
        else:
            plt.plot(p_range, 1-np.array(alg_crs[R]), linestyle='dotted', label='R = ' + R, marker='.')
    plt.xlabel("Probability of corruption")
    plt.ylabel("1 - Competitive Ratio")
    if title is not None:
        plt.title(title)
    plt.legend()
    if savefig:
        plt.savefig('{0}.png'.format(title), dpi=1000)
    plt.show()
    
    
def crs2(raw_data, keys=None):
    num_trials = len(raw_data)
    
    Rp_list = [Rp for Rp in raw_data[0]['unweighted'].keys() if Rp not in ['opt', 'advice', 'greedy', 'robust']]
    
    total_greedy = 0
    total_opt = 0
    total_advice = 0
    total_robust = 0
    total_ours = {}
    
    weight_keys = []
    if keys is None:
        weight_keys = raw_data[0].keys()
    else:
        weight_keys = keys
    num_weights = len(weight_keys)
    print(weight_keys)
    
    for trial in raw_data:
        for weight_key in weight_keys:
            d = trial[weight_key]
            opt = d['opt']
            total_greedy += d['greedy'] / opt
            total_advice += d['advice'] / opt
            total_robust += d['robust'] / opt
            for Rp in Rp_list:
                if Rp not in total_ours:
                    total_ours[Rp] = d[Rp]/opt
                else:
                    total_ours[Rp] += d[Rp]/opt
    for Rp in total_ours:
        total_ours[Rp] = total_ours[Rp]/(num_trials*num_weights)
    
    return total_ours, total_advice/(num_trials*num_weights), total_greedy/(num_trials*num_weights), total_robust/(num_trials*num_weights)
 


