
import os
import pickle
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import contextily as ctx
from shapely.geometry import Point, LineString
import datetime as dt
import time

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

def calculate_partial_time(edge_data, partial_length):
    full_length = edge_data['length']
    return (partial_length / full_length) * edge_data['time']

def speed_to_color(speed):
    if speed < 15:
        return 'orangered'
    elif speed < 30:
        return 'gold'
    else:
        return 'greenyellow'
def calculate_and_visualize_path(G, origin, destination, start_time, request_dir):
    origin_point = Point(origin[1], origin[0])
    destination_point = Point(destination[1], destination[0])


    origin_proj, origin_edge, origin_edge_data = match_point_to_edge(G, origin_point)
    dest_proj, dest_edge, dest_edge_data = match_point_to_edge(G, destination_point)
    

    origin_node = (origin_proj.x, origin_proj.y)
    dest_node = (dest_proj.x, dest_proj.y)
    G.add_node(origin_node, x=origin_proj.x, y=origin_proj.y)
    G.add_node(dest_node, x=dest_proj.x, y=dest_proj.y)


    origin_partial_length_0 = origin_proj.distance(Point(G.nodes[origin_edge[0]]['x'], G.nodes[origin_edge[0]]['y']))
    origin_partial_length_1 = origin_proj.distance(Point(G.nodes[origin_edge[1]]['x'], G.nodes[origin_edge[1]]['y']))
    dest_partial_length_0 = dest_proj.distance(Point(G.nodes[dest_edge[0]]['x'], G.nodes[dest_edge[0]]['y']))
    dest_partial_length_1 = dest_proj.distance(Point(G.nodes[dest_edge[1]]['x'], G.nodes[dest_edge[1]]['y']))

    G.add_edge(origin_edge[0], origin_node, length=origin_partial_length_0, time=calculate_partial_time(origin_edge_data, origin_partial_length_0))
    G.add_edge(origin_node, origin_edge[1], length=origin_partial_length_1, time=calculate_partial_time(origin_edge_data, origin_partial_length_1))
    G.add_edge(dest_edge[0], dest_node, length=dest_partial_length_0, time=calculate_partial_time(dest_edge_data, dest_partial_length_0))
    G.add_edge(dest_node, dest_edge[1], length=dest_partial_length_1, time=calculate_partial_time(dest_edge_data, dest_partial_length_1))


    route = nx.shortest_path(G, origin_node, dest_node, weight='time')
    route_length = nx.shortest_path_length(G, origin_node, dest_node, weight='length')
    route_time = nx.shortest_path_length(G, origin_node, dest_node, weight='time')

    route_points = [Point(G.nodes[node]['x'], G.nodes[node]['y']) for node in route]
    route_line = LineString(route_points)
    minx, miny, maxx, maxy = route_line.bounds

    # ����ͼ�����ĵ�ͱ߽�
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    lat_extent = maxy - miny
    lon_extent = maxx - minx
    padding = max(lat_extent, lon_extent) * 0.1  # ������ֵ�����ӻ�������ż���

    north = maxy + padding
    south = miny - padding
    east = maxx + padding
    west = minx - padding

    # ���ӻ�·��
    fig, ax = ox.plot_graph_route(G, route, route_color='orangered', node_size=0, bgcolor='white', edge_linewidth=1, show=False, close=False)
    
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        edge_data = G.get_edge_data(u, v)[0]
        if 'geometry' not in edge_data:
            point_u = Point(G.nodes[u]['x'], G.nodes[u]['y'])
            point_v = Point(G.nodes[v]['x'], G.nodes[v]['y'])
            edge_data['geometry'] = LineString([point_u, point_v])
        edge_geometry = edge_data['geometry']
        speed = (edge_data['length'] / edge_data['time']+1e-8) * 1.2 
        color = speed_to_color(speed)
        x, y = edge_geometry.xy
        ax.plot(x, y, color=color, linewidth=4)
    
    # ��ͼ�б�������յ�
    ax.scatter([origin[1]], [origin[0]], edgecolor='lightblue', facecolor='aqua', s=200, label='Origin', linewidth=3)
    ax.scatter([destination[1]], [destination[0]], edgecolor='violet', facecolor='fuchsia', s=200, label='Destination', linewidth=3)

    width=max(abs(east-west),abs(north-south))
    ax.set_xlim([(east+west-width)/2,(east+west+width)/2])
    ax.set_ylim([(north+south-width)/2, (north+south+width)/2])
    for _ in range(3):
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
        except:
            time.sleep(0.6)
    ax.set_xticks([])
    ax.set_yticks([])

    # ����ͼ�ĳ�������
    ax.set_aspect(aspect='equal')
    legend_elements = [Line2D([0], [0], color='orangered', lw=4, label='turtle craw'),
                    Line2D([0], [0], color='gold', lw=4, label='dog run'),
                    #Line2D([0], [0], color='yellow', lw=4, label='20-30 km/h'),
                    Line2D([0], [0], color='greenyellow', lw=4, label='horse run'),
                    Line2D([0], [0], marker='o', color='aqua', label='Origin', markersize=10, markerfacecolor='aqua', markeredgewidth=3),
                    Line2D([0], [0], marker='o', color='fuchsia', label='Destination', markersize=10, markerfacecolor='crimson', markeredgewidth=3)]
    ax.legend(handles=legend_elements)
    plt.savefig(os.path.join(request_dir,'route.png'), bbox_inches='tight')

    return route_length, route_time*attribute_bia + len(route)*0.2

def load_hourly_graph_files(output_dir='hourly_graphs'):
    hourly_graph_names = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))]
    threads = []
    GRAPH = dict()
    
    import threading
    def read_file(filename):
        with open(filename, 'rb') as file:
            idx = int(filename[-6:-4])
            GRAPH[str(idx)] = pickle.load(file)
            print(f"Read from {filename}")
            
    for filename in hourly_graph_names:
        thread = threading.Thread(target=read_file, args=(filename,))
        thread.start()
        threads.append(thread)
        
    for thread in threads:
        thread.join()
    return GRAPH, 1


def main(origin=None,
         destination=None,
         start_time=None,
         request_dir=None,
         hourly_graph_files=None,):


    G_hour = hourly_graph_files[str(start_time.hour)]

    route_length, route_time = calculate_and_visualize_path(G_hour, origin, destination, start_time, request_dir)

    return route_length, route_time

if __name__ == '__main__':
    output_dir = 'hourly_graphs'
    origin = (22.5382, 113.9787)  # 
    destination = (22.5333, 114.0557)  # 
    start_time = dt.datetime(2023, 4, 1, 15, 0)  #
    main(output_dir=output_dir,
         origin=origin,
         destination=destination,
         start_time=start_time,
         )