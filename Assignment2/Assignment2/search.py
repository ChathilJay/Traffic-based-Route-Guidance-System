import sys
from graph_parser import parse_problem_file
from search_algorithms import dfs, bfs, gbfs

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename, method = sys.argv[1], sys.argv[2].lower()
    nodes, graph, origin, goals = parse_problem_file(filename)

    if method == "dfs":
        goal, expanded, path = dfs(graph, origin, goals)
    elif method == "bfs":
        goal, expanded, path = bfs(graph, origin, goals)
    elif method == "gbfs":
        goal, expanded, path = gbfs(graph, nodes, origin, goals)
    else:
        print(f"Method '{method}' not implemented.")
        return

    print(f"{filename} {method}")
    if goal:
        print(goal, expanded)
        print(" -> ".join(map(str, path)))
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
