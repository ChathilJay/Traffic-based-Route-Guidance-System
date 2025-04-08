import os
import subprocess

methods = ["dfs", "bfs", "gbfs"]
files = [f for f in os.listdir("problem_files") if f.endswith(".txt")]

for f in sorted(files):
    for m in methods:
        print(f"\n=== Running {f} with {m} ===")
        subprocess.run(["python", "search.py", f"problem_files/{f}", m])
