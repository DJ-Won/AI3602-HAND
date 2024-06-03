#coding=gbk
import os
import pickle
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point, LineString
import datetime as dt
from matplotlib.lines import Line2D

# 定义函数将点匹配到最近的edge
def match_point_to_edge(G, point):
    nearest_edge_idx = ox.distance.nearest_edges(G, point.x, point.y, return_dist=True)
    edge = (nearest_edge_idx[0][0], nearest_edge_idx[0][1])  # (u, v)
    edge_data = G.get_edge_data(*edge)[0]  # 使用第一个键的边数据
    if 'geometry' not in edge_data:
        point_u = Point(G.nodes[nearest_edge_idx[0][0]]['x'], G.nodes[nearest_edge_idx[0][0]]['y'])
        point_v = Point(G.nodes[nearest_edge_idx[0][1]]['x'], G.nodes[nearest_edge_idx[0][1]]['y'])
        edge_data['geometry'] = LineString([point_u, point_v])
    edge_line = edge_data['geometry']
    projected_point = edge_line.interpolate(edge_line.project(point))
    return projected_point, edge, edge_data

# 定义函数计算部分路径时间
def calculate_partial_time(edge_data, partial_length):
    full_length = edge_data['length']
    return (partial_length / full_length) * edge_data['time']

# 定义速度范围到颜色的映射函数
def speed_to_color(speed):
    if speed < 15:
        return 'red'
    elif speed < 30:
        return 'orange'
    else:
        return 'green'

# 定义函数计算和可视化路径
def calculate_and_visualize_path(G, origin, destination, start_time):
    origin_point = Point(origin[1], origin[0])
    destination_point = Point(destination[1], destination[0])

    # 将起点和终点匹配到最近的edge上的某一点
    origin_proj, origin_edge, origin_edge_data = match_point_to_edge(G, origin_point)
    dest_proj, dest_edge, dest_edge_data = match_point_to_edge(G, destination_point)
    
    # 将起点和终点添加为新的节点
    origin_node = (origin_proj.x, origin_proj.y)
    dest_node = (dest_proj.x, dest_proj.y)
    G.add_node(origin_node, x=origin_proj.x, y=origin_proj.y)
    G.add_node(dest_node, x=dest_proj.x, y=dest_proj.y)

    # 添加新的边，从原始节点到起点/终点匹配的点
    origin_partial_length_0 = origin_proj.distance(Point(G.nodes[origin_edge[0]]['x'], G.nodes[origin_edge[0]]['y']))
    origin_partial_length_1 = origin_proj.distance(Point(G.nodes[origin_edge[1]]['x'], G.nodes[origin_edge[1]]['y']))
    dest_partial_length_0 = dest_proj.distance(Point(G.nodes[dest_edge[0]]['x'], G.nodes[dest_edge[0]]['y']))
    dest_partial_length_1 = dest_proj.distance(Point(G.nodes[dest_edge[1]]['x'], G.nodes[dest_edge[1]]['y']))

    G.add_edge(origin_edge[0], origin_node, length=origin_partial_length_0, time=calculate_partial_time(origin_edge_data, origin_partial_length_0))
    G.add_edge(origin_node, origin_edge[1], length=origin_partial_length_1, time=calculate_partial_time(origin_edge_data, origin_partial_length_1))
    G.add_edge(dest_edge[0], dest_node, length=dest_partial_length_0, time=calculate_partial_time(dest_edge_data, dest_partial_length_0))
    G.add_edge(dest_node, dest_edge[1], length=dest_partial_length_1, time=calculate_partial_time(dest_edge_data, dest_partial_length_1))

    # 计算从起点到终点的最快路径
    route = nx.shortest_path(G, origin_node, dest_node, weight='time')
    route_length = nx.shortest_path_length(G, origin_node, dest_node, weight='length')
    route_time = nx.shortest_path_length(G, origin_node, dest_node, weight='time')

    # 计算路径范围
    route_points = [Point(G.nodes[node]['x'], G.nodes[node]['y']) for node in route]
    route_line = LineString(route_points)
    minx, miny, maxx, maxy = route_line.bounds

    # 计算图的中心点和边界
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2
    lat_extent = maxy - miny
    lon_extent = maxx - minx
    padding = max(lat_extent, lon_extent) * 0.1  # 调整此值以增加或减少缩放级别

    north = maxy + padding
    south = miny - padding
    east = maxx + padding
    west = minx - padding

    # 创建静态地图
    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(len(route) - 1):
        u = route[i]
        v = route[i + 1]
        edge_data = G.get_edge_data(u, v)[0]
        if 'geometry' not in edge_data:
            point_u = Point(G.nodes[u]['x'], G.nodes[u]['y'])
            point_v = Point(G.nodes[v]['x'], G.nodes[v]['y'])
            edge_data['geometry'] = LineString([point_u, point_v])
        edge_geometry = edge_data['geometry']
        speed = float(edge_data['length'] / edge_data['time']) * 1.2 
        color = speed_to_color(speed)
        x, y = edge_geometry.xy
        ax.plot(x, y, color=color, linewidth=4)  # 增加路线宽度到 4

    # 标记起点和终点（空心圆点）
    ax.scatter([origin[1]], [origin[0]], edgecolor='lightblue', facecolor='blue', s=200, label='Origin', linewidth=3)
    ax.scatter([destination[1]], [destination[0]], edgecolor='yellow', facecolor='gold', s=200, label='Destination', linewidth=3)

    

    # 设置视图边界
    width=max(abs(east-west),abs(north-south))
    ax.set_xlim([(east+west-width)/2,(east+west+width)/2])
    ax.set_ylim([(north+south-width)/2, (north+south+width)/2])
    # 添加背景地图
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik)
    # 移除经纬度框
    ax.set_xticks([])
    ax.set_yticks([])

    # 设置图的长宽比例
    ax.set_aspect(aspect='equal')

    # 添加图例，包括起点和终点
    legend_elements = [Line2D([0], [0], color='red', lw=4, label='拥堵'),
                       Line2D([0], [0], color='orange', lw=4, label='缓行'),
                       #Line2D([0], [0], color='yellow', lw=4, label='20-30 km/h'),
                       Line2D([0], [0], color='green', lw=4, label='畅行'),
                       Line2D([0], [0], marker='o', color='blue', label='出发点', markersize=10, markerfacecolor='blue', markeredgewidth=3),
                       Line2D([0], [0], marker='o', color='gold', label='目的地', markersize=10, markerfacecolor='gold', markeredgewidth=3)]
    ax.legend(handles=legend_elements)

    plt.show()

    return route_length, route_time * 3.2

# 读取每小时生成的图文件
output_dir = 'hourly_graphs'
hourly_graph_files = [os.path.join(output_dir, f) for f in sorted(os.listdir(output_dir))]

# 输入起点与终点的经纬度与出发时间
origin = (22.5382, 113.9787)  # 起点经纬度
destination = (22.5333, 114.0557)  # 终点经纬度
start_time = dt.datetime(2023, 4, 1, 15, 0)  # 出发时间

# 根据出发时间选择对应时间段的图
hour = start_time.hour
with open(hourly_graph_files[hour], 'rb') as f:
    G_hour = pickle.load(f)

# 计算路径并可视化
route_length, route_time = calculate_and_visualize_path(G_hour, origin, destination, start_time)

print(f"Total Distance: {route_length} meters")
print(f"Total Time: {route_time} seconds")