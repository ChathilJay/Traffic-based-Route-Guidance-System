# search.py

import sys

def parse_problem_file(filename):
    nodes = {}
    graph = {}
    origin = None
    destinations = []

    with open(filename, 'r') as f:
        section = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                section = "origin"
                continue
            elif line.startswith("Destinations:"):
                section = "destinations"
                continue

            if section == "nodes":
                node_id, coord = line.split(":")
                node_id = int(node_id.strip())
                x, y = eval(coord.strip())
                nodes[node_id] = (x, y)
                graph[node_id] = {}
            elif section == "edges":
                edge_part, cost = line.split(":")
                node_from, node_to = eval(edge_part.strip())
                cost = int(cost.strip())
                if node_from not in graph:
                    graph[node_from] = {}
                graph[node_from][node_to] = cost
            elif section == "origin":
                origin = int(line.strip())
            elif section == "destinations":
                destinations = [int(x.strip()) for x in line.split(";")]

    return nodes, graph, origin, destinations

def dfs(graph, origin, destinations):
    stack = [(origin, [origin])]  # (current_node, path_so_far)
    visited = set()
    nodes_expanded = 0

    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current in destinations:
            return current, nodes_expanded, path

        neighbors = sorted(graph.get(current, {}).keys(), reverse=True)
        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return None, nodes_expanded, []  # No path found

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        sys.exit(1)

    filename = sys.argv[1]
    method = sys.argv[2].lower()

    nodes, graph, origin, destinations = parse_problem_file(filename)

    if method == "dfs":
        goal, num_nodes, path = dfs(graph, origin, destinations)
    else:
        print(f"Method '{method}' not implemented yet.")
        sys.exit(1)

    print(f"{filename} {method}")
    if goal:
        print(goal, num_nodes)
        print(" -> ".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
