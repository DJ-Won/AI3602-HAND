#coding=gbk
import osmnx as ox
import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
from scipy.spatial import cKDTree

filename = 'graph_shenzhen.pkl'
data_files = ['9.csv', '10.csv', '11.csv', '12.csv', '13.csv']#cleaned original data of shenzhen taxis
output_dir = 'hourly_graphs'

# create output dir for 12 h graphs
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# first time create ,then load 
if os.path.exists(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    print("Graph loaded from file.")
else:
    G = ox.graph_from_address('深圳', dist=100000, network_type='drive')
    with open(filename, 'wb') as f:
        pickle.dump(G, f)
    print("Graph saved to file.")

# KDTree for matching the nearest edges
def prepare_kdtree(G):
    edge_points = []
    edge_idx = []
    for u, v, key, data in G.edges(keys=True, data=True):
        point_u = (G.nodes[u]['x'], G.nodes[u]['y'])
        point_v = (G.nodes[v]['x'], G.nodes[v]['y'])
        edge_points.append(point_u)
        edge_points.append(point_v)
        edge_idx.append((u, v))
    edge_kdtree = cKDTree(edge_points)
    return edge_kdtree, edge_idx

edge_kdtree, edge_idx = prepare_kdtree(G)

def find_nearest_edge(edge_kdtree, edge_idx, point):
    distance, index = edge_kdtree.query(point)
    nearest_edge = edge_idx[index // 2]
    return nearest_edge

# read all csv 
def read_all_data(data_files):
    data = []
    for file in data_files:
        df = pd.read_csv(file)
        data.append(df)
    return pd.concat(data, ignore_index=True)

all_data = read_all_data(data_files)

# process data and create graph for each hour
for hour in tqdm(range(24), desc="Processing hours"):
    hourly_data = all_data[all_data['hour'] == hour]
    if hourly_data.empty:
        continue
    
    hourly_data = hourly_data[['经度', '纬度', '速度']]
    hourly_data['经度'] = hourly_data['经度'].astype(float)
    hourly_data['纬度'] = hourly_data['纬度'].astype(float)
    hourly_data['速度'] = hourly_data['速度'].astype(float)
    
    current_hour_speeds = hourly_data['速度'][hourly_data['速度'] > 0].tolist()
    global_avg_speed = np.mean(current_hour_speeds) if current_hour_speeds else 1.0
    mean_speed = np.mean(current_hour_speeds) if current_hour_speeds else 1.0

    G_hour = G.copy()
    
    for u, v, key, data in G_hour.edges(keys=True, data=True):
        data['time'] = None
        data['speeds'] = []

    points = hourly_data[['经度', '纬度']].values
    speeds = hourly_data['速度'].values
    nearest_edges = [find_nearest_edge(edge_kdtree, edge_idx, point) for point in points]

    for (u, v), speed in zip(nearest_edges, speeds):
        if speed > 0:
            G_hour[u][v][0]['speeds'].append(speed)

    for u, v, key, data in G_hour.edges(keys=True, data=True):
        if 'speeds' in data and data['speeds']:
            avg_speed = np.mean(data['speeds'])
        else:
            avg_speed = global_avg_speed
        
        if avg_speed > 0:
            travel_time = data['length'] / avg_speed
        else:
            travel_time = data['length'] / mean_speed
        
        data['time'] = travel_time
        del data['speeds']
    
    hour_file = os.path.join(output_dir, f'graph_shenzhen_hour_{hour:02d}.pkl')
    with open(hour_file, 'wb') as f:
        pickle.dump(G_hour, f)
    
    print(f"Graph for hour {hour} saved to {hour_file}")

print("All hourly graphs have been processed and saved.")
