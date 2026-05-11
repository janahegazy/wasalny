
# Public Transportation Optimization using Dynamic Programming
# Goal:
# 1) Optimize bus + metro schedules
# 2) Allocate transportation resources efficiently
# 3) Build integrated transportation network
# 4) Optimize transfer points


import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ============================================================
# PART 1: DEFINE TRANSPORT NETWORK
# ============================================================

# Graph:
# Nodes = Stations
# Edges = Routes with travel times

G = nx.DiGraph()

stations = [
    "A", "B", "C", "D", "E", "F"
]

# Add routes:
# (from, to, travel_time, mode)
routes = [
    ("A", "B", 10, "Bus"),
    ("A", "C", 15, "Metro"),
    ("B", "D", 12, "Bus"),
    ("C", "D", 5, "Metro"),
    ("B", "E", 20, "Bus"),
    ("D", "E", 8, "Metro"),
    ("D", "F", 10, "Bus"),
    ("E", "F", 5, "Metro")
]

for u, v, time, mode in routes:
    G.add_edge(u, v, weight=time, mode=mode) #constructing the graph

# PART 2: DYNAMIC PROGRAMMING FOR SCHEDULING
# DP[node] = Minimum travel time from source

def optimize_schedule(graph, source):
    dp = {node: float('inf') for node in graph.nodes} #i initialized dictionary for dp and made every node = infinity bec. i don't know the nodes yet
    dp[source] = 0  # i initialized the starting point=0 bec. i don't know the start yet 
    parent = {}   #to know what is the parent node of the new node to return the full path at the end

    # Relax edges repeatedly (Bellman-Ford style DP)
    for _ in range(len(graph.nodes) - 1):    #in range of edges bec. num of edges=n-1 ,_ means We need to repeat this loop several times, but we do NOT care about the loop variable itself.
        for u, v, data in graph.edges(data=True):
            if dp[u] + data['weight'] < dp[v]:    #we check if the current path is faster than the next path
                dp[v] = dp[u] + data['weight']
                parent[v] = u

    return dp, parent

# PART 3: RESOURCE ALLOCATION USING DP (Knapsack Style)
# Goal: Maximize coverage ( passengers ) with limited buses

def allocate_buses(route_demands, total_buses):
    n = len(route_demands)

    # DP table
    dp = [[0 for _ in range(total_buses + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        demand = route_demands[i - 1]

        for buses in range(total_buses + 1):
            dp[i][buses] = dp[i - 1][buses]

            # allocate x buses
            for x in range(1, buses + 1):
                coverage = min(demand, x * 10)  # each bus serves 10 passengers
                dp[i][buses] = max(
                    dp[i][buses],
                    dp[i - 1][buses - x] + coverage
                )

    return dp[n][total_buses]

# ============================================================
# PART 4: TRANSFER POINT OPTIMIZATION
# Penalty added when changing transport mode
# ============================================================

def optimize_with_transfer(graph, source):
    dp = {node: float('inf') for node in graph.nodes}
    dp[source] = 0

    for _ in range(len(graph.nodes) - 1):
        for u, v, data in graph.edges(data=True):
            transfer_penalty = 0

            # Check predecessor mode changes
            for pred in graph.predecessors(u):
                if graph[pred][u]['mode'] != data['mode']:
                    transfer_penalty = 3  # 3 min transfer time

            if dp[u] + data['weight'] + transfer_penalty < dp[v]:
                dp[v] = dp[u] + data['weight'] + transfer_penalty

    return dp

# ============================================================
# PART 5: VISUALIZATION
# ============================================================

def visualize_network(graph):
    pos = nx.spring_layout(graph)

    edge_labels = {
        (u, v): f"{d['mode']} ({d['weight']} min)"
        for u, v, d in graph.edges(data=True)
    }

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=2500,
        node_color="lightblue",
        font_size=10
    )

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8
    )

    plt.title("Integrated Public Transportation Network")
    plt.show()

# ============================================================
# PART 6: PERFORMANCE ANALYSIS
# ============================================================

def analyze_performance(before, after):
    data = pd.DataFrame({
        "Station": list(before.keys()),
        "Before Optimization": list(before.values()),
        "After Optimization": list(after.values())
    })

    data["Improvement"] = (
        data["Before Optimization"] - data["After Optimization"]
    )

    print("\n=== Travel Time Improvement Analysis ===")
    print(data)

# ============================================================
# MAIN EXECUTION
# ============================================================

source_station = "A"

# Schedule optimization
before_dp, _ = optimize_schedule(G, source_station)

# Transfer-aware optimization
after_dp = optimize_with_transfer(G, source_station)

# Resource allocation
route_demands = [50, 40, 60, 30]  # passenger demand
total_buses = 10

max_coverage = allocate_buses(route_demands, total_buses)

# ============================================================
# OUTPUT
# ============================================================

print("=== Optimized Travel Times (Without Transfer Optimization) ===")
for station, time in before_dp.items():
    print(f"{source_station} -> {station}: {time} min")

print("\n=== Optimized Travel Times (With Transfer Optimization) ===")
for station, time in after_dp.items():
    print(f"{source_station} -> {station}: {time} min")

print(f"\n=== Maximum Passenger Coverage with {total_buses} buses ===")
print(f"Coverage: {max_coverage} passengers")

# Visualization
visualize_network(G)

# Analysis
analyze_performance(before_dp, after_dp)
