# cpu frequency, temperature, throttle when all cores are loaded

import psutil, time, csv
import numpy as np

logfile='cpu_baseline_log.csv'
fields = ['timestamp', 'cpu_freq_mhz', 'cpu_percent', 'cpu_temp']
latencies = []

with open(logfile, 'w', newline='') as f: 
    writer = csv.writer(f)
    writer.writerow(fields)


    for _ in range(300): # 5 min at 1 Hz, adjust if I adjust the stress testing time
        freq = psutil.cpu_freq().current
        load = psutil.cpu_percent()
        temp = psutil.sensors_temperatures()['coretemp'][0].current
        writer.writerow([time.time(), freq, load, temp])
        start = time.perf_counter()
        # trivial CPU workload
        sum(range(1000))
        end = time.perf_counter()
        latencies.append(end - start)
        time.sleep(1)


latencies = np.array(latencies)
print(f"Mean latency: {latencies.mean()*1e6:.2f} µs")
print(f"Std deviation: {latencies.std()*1e6:.2f} µs")