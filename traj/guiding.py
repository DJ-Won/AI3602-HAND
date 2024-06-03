#coding=gbk
import os
import pickle
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import datetime as dt

# ���庯������ƥ�䵽�����edge
def match_point_to_edge(G, point):
    nearest_edge_idx = ox.distance.nearest_edges(G, point.x, point.y, return_dist=True)
    edge = (nearest_edge_idx[0][0], nearest_edge_idx[0][1])  # (u, v)
    edge_data = G.get_edge_data(*edge)[0]  # ʹ�õ�һ�����ı�����
    #print(G.nodes[nearest_edge_idx[0][0]])
    if 'geometry' not in edge_data:
        point_u = Point(G.nodes[nearest_edge_idx[0][0]]['x'], G.nodes[nearest_edge_idx[0][0]]['y'])
        point_v = Point(G.nodes[nearest_edge_idx[0][1]]['x'], G.nodes[nearest_edge_idx[0][1]]['y'])
        edge_data['geometry'] = LineString([point_u, point_v])
    edge_line = edge_data['geometry']
    projected_point = edge_line.interpolate(edge_line.project(point))
    return projected_point, edge, edge_data
attribute_bia=3.2
# ���庯�����㲿��·��ʱ��
def calculate_partial_time(edge_data, partial_length):
    full_length = edge_data['length']
    return (partial_length / full_length) * edge_data['time']

# ���庯������Ϳ��ӻ�·��
def calculate_and_visualize_path(G, origin, destination, start_time):
    origin_point = Point(origin[1], origin[0])
    destination_point = Point(destination[1], destination[0])

    # �������յ�ƥ�䵽�����edge�ϵ�ĳһ��
    origin_proj, origin_edge, origin_edge_data = match_point_to_edge(G, origin_point)
    dest_proj, dest_edge, dest_edge_data = match_point_to_edge(G, destination_point)
    
    # �������յ����Ϊ�µĽڵ�
    origin_node = (origin_proj.x, origin_proj.y)
    dest_node = (dest_proj.x, dest_proj.y)
    G.add_node(origin_node, x=origin_proj.x, y=origin_proj.y)
    G.add_node(dest_node, x=dest_proj.x, y=dest_proj.y)

    # ����µıߣ���ԭʼ�ڵ㵽���/�յ�ƥ��ĵ�
    origin_partial_length_0 = origin_proj.distance(Point(G.nodes[origin_edge[0]]['x'], G.nodes[origin_edge[0]]['y']))
    origin_partial_length_1 = origin_proj.distance(Point(G.nodes[origin_edge[1]]['x'], G.nodes[origin_edge[1]]['y']))
    dest_partial_length_0 = dest_proj.distance(Point(G.nodes[dest_edge[0]]['x'], G.nodes[dest_edge[0]]['y']))
    dest_partial_length_1 = dest_proj.distance(Point(G.nodes[dest_edge[1]]['x'], G.nodes[dest_edge[1]]['y']))

    G.add_edge(origin_edge[0], origin_node, length=origin_partial_length_0, time=calculate_partial_time(origin_edge_data, origin_partial_length_0))
    G.add_edge(origin_node, origin_edge[1], length=origin_partial_length_1, time=calculate_partial_time(origin_edge_data, origin_partial_length_1))
    G.add_edge(dest_edge[0], dest_node, length=dest_partial_length_0, time=calculate_partial_time(dest_edge_data, dest_partial_length_0))
    G.add_edge(dest_node, dest_edge[1], length=dest_partial_length_1, time=calculate_partial_time(dest_edge_data, dest_partial_length_1))

    # �������㵽�յ�����·��
    route = nx.shortest_path(G, origin_node, dest_node, weight='time')
    route_length = nx.shortest_path_length(G, origin_node, dest_node, weight='length')
    route_time = nx.shortest_path_length(G, origin_node, dest_node, weight='time')

    # ����ͼ�����ĵ�ͱ߽�
    latitudes = [origin[0], destination[0]]
    longitudes = [origin[1], destination[1]]
    center_lat = np.mean(latitudes)
    center_lon = np.mean(longitudes)
    padding = 0.03  # ������ֵ�����ӻ�������ż���

    north = max(latitudes) + padding
    south = min(latitudes) - padding
    east = max(longitudes) + padding
    west = min(longitudes) - padding

    # ���ӻ�·��
    fig, ax = ox.plot_graph_route(G, route, route_color='red', node_size=0, bgcolor='white', edge_linewidth=1, show=False, close=False)
    
    # ������ͼ�߽�
    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    
    # ��ͼ�б�������յ�
    ax.scatter(origin[1], origin[0], c='blue', s=100, label='Origin')
    ax.scatter(destination[1], destination[0], c='green', s=100, label='Destination')
    ax.legend()
    
    plt.show()

    return route_length, route_time*attribute_bia

# ��ȡÿСʱ���ɵ�ͼ�ļ�
output_dir = 'hourly_graphs'
hourly_graph_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))]

# ����������յ�ľ�γ�������ʱ��
origin = (22.5382,  113.9787)  # ��㾭γ��
destination = (22.5333, 114.0557)  # �յ㾭γ��
start_time = dt.datetime(2023, 4, 1, 15, 0)  # ����ʱ��

# ���ݳ���ʱ��ѡ���Ӧʱ��ε�ͼ
hour = start_time.hour
with open(hourly_graph_files[hour], 'rb') as f:
    G_hour = pickle.load(f)

# ����·�������ӻ�
route_length, route_time = calculate_and_visualize_path(G_hour, origin, destination, start_time)

print(f"Total Distance: {route_length} meters")
print(f"Total Time: {route_time} seconds")
