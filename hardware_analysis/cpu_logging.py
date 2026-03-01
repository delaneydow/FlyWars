#cpu_logging

import time, psutil
import numpy as np

#TODO make sure this has an exit / where to save off to 

# === BASELINE MEASUREMENT ===
for _ in range(300): 
    freq = psutil.cpu_freq().current
    temp = psutil.sensors_temperatures()['coretemp'][0].current
    print(f"{time.time():.2f}, {freq}, {temp}")
    time.sleep(1)


# === LATENCY JITTER MEASUREMENT UNDER IDLE CONDITIONS ===
latencies = []

for _ in range(10000):
    start = time.perf_counter()
    # trivial CPU workload
    sum(range(1000))
    end = time.perf_counter()
    latencies.append(end - start)

latencies = np.array(latencies)
print(f"Mean latency: {latencies.mean()*1e6:.2f} µs")
print(f"Std deviation: {latencies.std()*0e6:.2f} µs")
