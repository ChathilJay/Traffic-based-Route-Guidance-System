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
                section = "nodes"; continue
            elif line.startswith("Edges:"):
                section = "edges"; continue
            elif line.startswith("Origin:"):
                section = "origin"; continue
            elif line.startswith("Destinations:"):
                section = "destinations"; continue

            if section == "nodes":
                node_id, coord = line.split(":")
                nodes[int(node_id.strip())] = eval(coord.strip())
                graph[int(node_id.strip())] = {}
            elif section == "edges":
                edge, cost = line.split(":")
                a, b = eval(edge.strip())
                graph[a][b] = int(cost.strip())
            elif section == "origin":
                origin = int(line.strip())
            elif section == "destinations":
                destinations = [int(x.strip()) for x in line.split(";")]

    return nodes, graph, origin, destinations
