#coding=gbk
import osmnx as ox
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import folium


output_dir = 'hourly_graphs'


def classify_edges_by_speed(G):
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'length' in data and 'time' in data and data['time'] > 0:
            speed = data['length'] / data['time']
            if speed <= 15:
                G[u][v][key]['congestion'] = 'congested'
                G[u][v][key]['color'] = 'orangered'
            elif 15 < speed <= 25:
                G[u][v][key]['congestion'] = 'moderately free'
                G[u][v][key]['color'] = 'gold'
            elif 25 < speed <= 35:
                G[u][v][key]['congestion'] = 'free'
                G[u][v][key]['color'] = 'greenyellow'
            else:
                G[u][v][key]['congestion'] = 'very free'
                G[u][v][key]['color'] = 'deepskyblue'

def visualize_graph(G, hour):
    fig, ax = plt.subplots(figsize=(10, 10))
    edge_colors = [data['color'] for u, v, key, data in G.edges(keys=True, data=True)]
    ox.plot_graph(G, ax=ax, edge_color=edge_colors, edge_linewidth=2, node_size=0, bgcolor='k')
    plt.title(f'Traffic Congestion for Hour {hour}')
    plt.show()

def visualize_graph_on_map(G, hour):
    G_proj = ox.project_graph(G, to_crs='EPSG:4326')
    m = folium.Map(location=[31.23, 121.47], zoom_start=16) 
    for u, v, key, data in G_proj.edges(keys=True, data=True):
        color = data['color']
        points = [(G_proj.nodes[u]['y'], G_proj.nodes[u]['x']), (G_proj.nodes[v]['y'], G_proj.nodes[v]['x'])]
        folium.PolyLine(points, color=color, weight=5).add_to(m)
    
    folium.LayerControl().add_to(m)
    m.save(f'traffic_congestion_hour_{hour}.html')

#Here to change the time of a day
Time=22
with open(f"hourly_graphs\graph_shenzhen_hour_"+str(Time)+".pkl", 'rb') as f:
    G = pickle.load(f)

classify_edges_by_speed(G)
visualize_graph(G, Time)
visualize_graph_on_map(G,Time)

print("All hourly graphs have been processed and visualized.")
