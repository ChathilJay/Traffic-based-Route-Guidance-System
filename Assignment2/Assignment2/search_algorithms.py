from collections import deque
import heapq

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

def heuristic(coords, node, goals):
    x1, y1 = coords[node]
    return min((x1 - x2)**2 + (y1 - y2)**2 for (x2, y2) in [coords[g] for g in goals])

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
