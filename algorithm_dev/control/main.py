import pandas as pd
import time
import signal
import sys

from algorithm_dev.control.control_interface import control_step
from algorithm_dev.control.cooldown import *

# import Tracking pipeline
from algorithm_dev.vision.track import Track #import Track class, TODO see if track needs to be passed/accessed
from algorithm_dev.vision.tracking import process_video
from algorithm_dev.control.laser_interface import LaserInterface
from algorithm_dev.control.mirror_planner import MirrorPlanner
from algorithm_dev.Writer import Writer

# === latency settings ===
FRAME_DT = 1/120.0
TOTAL_LATENCY = 0.075 # seconds (initial estimate)
HORIZON_RANGE = range(4,13) # frames to sweep  #TODO how does it know how many horizons it needs? 
PAUSED = False
laser = None
mirror = None

def toggle_pause(signum, frame): 
    global PAUSED, laser, mirror

    PAUSED = not PAUSED
    print(f"[SYSTEM] Paused = {PAUSED}")

    if PAUSED:
        if laser:
            laser.off()
        if mirror:
            mirror.off()


def emergency_stop(signum, frame): 
    global laser, mirror

    print("\n[EMERGENCY STOP]")

    if laser:
        laser.off()
        laser.ser.close()

    if mirror:
        mirror.off()
        mirror.close()



# main control/ pipeline
def main(): 

    global mirror, laser 
    laser = LaserInterface()
    mirror = MirrorPlanner() 

    signal.signal(signal.SIGINT, emergency_stop) #CTRL+C is controlled shutdown 
    signal.signal(signal.SIGUSR1, toggle_pause) #PID from ps aux | grep main.py, kill / resume -USR1 <PID>


    writer = Writer() 
    # === CALL VISION LOOP ===
    
    try: 

        last_packet_time = time.perf_counter()
        WATCHDOG_TIMEOUT = 0.1 #100 ms, deadman watchdog 
    
        for packet in process_video(): 

            now = time.perf_counter()

            if now - last_packet_time > WATCHDOG_TIMEOUT:
                print("[WATCHDOG] Vision timeout")
                if laser:
                    laser.off()
                if mirror: 
                    mirror.off()

            last_packet_time = now #system kills on lag 

            while PAUSED:
                if laser: 
                    laser.off()
                if mirror:
                    mirror.off()
                time.sleep(0.1)


            # run control
            pipeline_start = packet["timestamp"] # starting time 
            result = control_step(packet["tracks"],
                                packet["states"],
                                packet["frame"]) 

            #thermal monitoring
            cpu_temp = get_cpu_temp()
            cooldown = adaptive_cooldown(cpu_temp)

            # TODO figure out if I should send cooldown to planning or not? 
            if cooldown > 0: 
                time.sleep(cooldown)

            total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000 #ms

            writer.log_frame({
                "frame": packet["frame"],
                "time": time.time(),
                "vision_latency": packet["vision_latency_ms"],
                "pipeline_latency": total_pipeline_ms,
                "ndet": packet["detections"],
                "ntrack": len(packet["tracks"]),
                "temp": cpu_temp,
                "cooldown": cooldown
            })

            writer.log_fire(result)

        #TODO maybe df of quick/final stats (i.e. save off total fire_count among other variables)     

    #TODO maybe occasionally print updates
    finally: 
        emergency_stop(None, None) #shutdown as default 
        print ("emergency stop completed.")
        sys.exit(0)

if __name__=="__main__": 
        main()
