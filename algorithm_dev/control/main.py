import pandas as pd
import time

from control_interface import control_step
from cooldown import *

# import Tracking pipeline
from algorithm_dev.vision.track import Track #import Track class, TODO see if track needs to be passed/accessed
from algorithm_dev.vision.tracking import process_video
from algorithm_dev.Writer import Writer

# === latency settings ===
FRAME_DT = 1/120.0
TOTAL_LATENCY = 0.075 # seconds (initial estimate)
HORIZON_RANGE = range(4,13) # frames to sweep  #TODO how does it know how many horizons it needs? 



# main control/ pipeline
def main(): 
    writer = Writer() 
    # === CALL VISION LOOP ===
    
    
    for packet in process_video(): 

        # run control
        start = time.perf_counter()
        result = control_step(packet["tracks"],
                              packet["states"],
                              packet["frame"]) 
        t_cntrl = (time.perf_counter()- start) * 1000 #ms

        #thermal monitoring
        cpu_temp = get_cpu_temp()
        cooldown = adaptive_cooldown(cpu_temp)

        # TODO figure out if I should send cooldown to planning or not? 
        if cooldown > 0: 
            time.sleep(cooldown)

        total_pipeline = (time.perf_counter() - start) * 1000 #ms

        writer.log_frame({
            "frame": packet["frame"],
            "time": time.time(),
            "vision_latency": packet["latency_ms"],
            "ndet": packet["detections"],
            "ntrack": len(packet["tracks"]),
            "temp": cpu_temp,
            "cooldown": cooldown
        })

        writer.log_fire(result)

    #TODO maybe df of quick/final stats (i.e. save off total fire_count among other variables)     

   #TODO maybe occasionally print updates

if __name__=="__main__": 
    main()
