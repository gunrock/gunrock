# This is an example program to collect metrics from an algorithm.
# It should be run from the build directory.
# Most metrics are collected with the compile flag ESSENTIALS_COLLECT_METRICS=ON.
# Runtimes need to be collected WITHOUT this flag enabled.
# This script compiles and runs both versions of the algorithm and 
# exports the runtimes and other statistics from the respective versions.
# It then combines the outputs, using all metrics from the version
# with ESSENTIALS_COLLECT_METRICS=ON except for runtimes. 
# Anomalous runs are filtered out, summary statistics are updated, and MTEPS are
# recalculated with the appropriate runtimes.
import json
import subprocess
from statistics import mean, stdev

# Input parameters here
output_filename = "test"
algorithm = "sssp"
dataset = "../datasets/chesapeake/chesapeake.mtx"
sources = "0,0,0,0,0"

# Build and run
subprocess.run(["cmake", "-DESSENTIALS_COLLECT_METRICS=ON", ".."])
subprocess.run(["make", algorithm])
subprocess.run(["mv", "./bin/" + algorithm, "./bin/" + algorithm + "_metrics"])
subprocess.run(["cmake", "-DESSENTIALS_COLLECT_METRICS=OFF", ".."])
subprocess.run(["make", algorithm])

subprocess.run(["./bin/" + algorithm + "_metrics", "-m", dataset, "-s", sources, 
                "--export_metrics", "-f", output_filename + "_metrics"])
subprocess.run(["./bin/" + algorithm, "-m", dataset, "-s", sources, "--export_metrics", 
                "-f", output_filename + "_runtimes"])

output = {}

# Get data
metrics_content = open(output_filename + "_metrics")
metrics_data = json.load(metrics_content)

# Add all metrics but runtimes and mteps to output 
for key in metrics_data.keys():
    if "process_time" not in key and "mteps" not in key:
        output[key] = metrics_data[key]

metrics_content.close()

# Get runtimes
runtimes_content = open(output_filename + "_runtimes")
runtimes_data = json.load(runtimes_content)

# Only add the runtimes to output 
for key in runtimes_data.keys():
    if "process_time" in key:
        output[key] = runtimes_data[key]

runtimes_content.close()

# Filter out runs with runtimes more than 2 standard deviations from the average
output["filtered_process_times"] = []
output["filtered_edges_visited"] = []
output["filtered_nodes_visited"] = []
output["filtered_search_depths"] = []
output["filtered_sources"] = []

for i in range(len(output["process_times"])):
    if abs(output["avg_process_time"] - output["process_times"][i]) <= (2 * output["stddev_process_time"]):
        output["filtered_process_times"].append(output["process_times"][i])
        output["filtered_edges_visited"].append(output["edges_visited"][i])
        output["filtered_nodes_visited"].append(output["nodes_visited"][i])
        output["filtered_search_depths"].append(output["search_depths"][i])
        if len(output["srcs"]) > i:
            output["filtered_sources"].append(output["srcs"][i])

# Recalculate summary statistics with filtered runs only
if len(output["process_times"]) != len(output["filtered_process_times"]):
    output["avg_process_time"] = mean(output["filtered_process_times"])
    output["max_process_time"] = max(output["filtered_process_times"])
    output["min_process_time"] = min(output["filtered_process_times"])
    output["stddev_process_time"] = stdev(output["filtered_process_times"])
    
    output["avg_search_depth"] = mean(output["filtered_search_depths"])
    output["max_search_depth"] = max(output["filtered_search_depths"])
    output["min_search_depth"] = min(output["filtered_search_depths"])

# Calculate mteps with updated runtimes
mteps = [float(i) / (j * 1000) for i, j in zip(output["filtered_edges_visited"], output["filtered_process_times"])]
avg_mteps = mean(mteps)
min_mteps = min(mteps)
max_mteps = max(mteps)

output["mteps"] = mteps
output["avg_mteps"] = avg_mteps
output["max_mteps"] = max_mteps
output["min_mteps"] = min_mteps

# Write data to file
outfile = open(output_filename, "w")
outfile.write(json.dumps(output, indent=4, sort_keys=True))
outfile.close()
