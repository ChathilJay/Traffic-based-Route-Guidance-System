# parser.py

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
                graph[node_id] = {}  # initialize adjacency list
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

nodes, graph, origin, destinations = parse_problem_file("PathFinder-test.txt")
print("Nodes:", nodes)
print("Graph:", graph)
print("Origin:", origin)
print("Destinations:", destinations)