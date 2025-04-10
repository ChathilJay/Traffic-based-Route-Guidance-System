from collections import deque
import heapq
import math

#Depth First Search
def dfs(graph, origin, destinations):
    stack, visited, expanded = [(origin, [origin])], set(), 0
    while stack:
        node, path = stack.pop()
        if node in visited: continue
        visited.add(node); expanded += 1
        if node in destinations: return node, expanded, path
        for neighbor in sorted(graph[node], reverse=True):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None, expanded, []

#Breadth First Seacrh
def bfs(graph, origin, destinations):
    queue, visited, expanded = deque([(origin, [origin])]), set(), 0
    while queue:
        node, path = queue.popleft()
        if node in visited: continue
        visited.add(node); expanded += 1
        if node in destinations: return node, expanded, path
        for neighbor in sorted(graph[node]):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None, expanded, []


#def heuristic(coords, node, goals):
    x1, y1 = coords[node]
    return min(math.sqrt((x1 - x2)**2 + (y1 - y2)**2) for (x2, y2) in [coords[g] for g in goals])

def heuristic(coords, node, goals):
    x1, y1 = coords[node]
    return min((x1 - x2)**2 + (y1 - y2)**2 for (x2, y2) in [coords[g] for g in goals])


#Greedy best first search
def gbfs(graph, coords, origin, destinations):
    heap, visited, expanded = [(heuristic(coords, origin, destinations), origin, [origin])], set(), 0
    while heap:
        _, node, path = heapq.heappop(heap)
        if node in visited: continue
        visited.add(node); expanded += 1
        if node in destinations: return node, expanded, path
        for neighbor in sorted(graph[node]):
            if neighbor not in visited:
                h = heuristic(coords, neighbor, destinations)
                heapq.heappush(heap, (h, neighbor, path + [neighbor]))
    return None, expanded, []

#A star search
def a_star(graph, coords, origin, destinations):
    heap = []
    heapq.heappush(heap, (0 + heuristic(coords, origin, destinations), 0, origin, [origin]))
    visited = set()
    cost_so_far = {origin: 0}
    nodes_expanded = 0

    while heap:
        f_score, g, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current in destinations:
            return current, nodes_expanded, path

        for neighbor, edge_cost in graph[current].items():
            new_cost = g + edge_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                f = new_cost + heuristic(coords, neighbor, destinations)
                heapq.heappush(heap, (f, new_cost, neighbor, path + [neighbor]))

    return None, nodes_expanded, []

#Custom Search 1 (DLS - Depth Limited search)
def cus1(graph, origin, destinations, max_depth=4):
    stack = [(origin, [origin], 0)]
    visited = set()
    nodes_expanded = 0

    while stack:
        current, path, depth = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current in destinations:
            return current, nodes_expanded, path

        if depth < max_depth:
            for neighbor in sorted(graph[current], reverse=True):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], depth + 1))

    return None, nodes_expanded, []

#Custom search 2 
def cus2(graph, coords, origin, destinations):
    heap = []
    heapq.heappush(heap, (0 + heuristic(coords, origin, destinations), 0, origin, [origin]))
    visited = set()
    nodes_expanded = 0

    while heap:
        f_score, steps, current, path = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current in destinations:
            return current, nodes_expanded, path

        for neighbor in sorted(graph[current]):
            if neighbor not in visited:
                new_steps = steps + 1
                f = new_steps + heuristic(coords, neighbor, destinations)
                heapq.heappush(heap, (f, new_steps, neighbor, path + [neighbor]))

    return None, nodes_expanded, []



