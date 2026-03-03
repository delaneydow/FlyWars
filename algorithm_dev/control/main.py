import pandas as pd
import time
import signal
import sys

from algorithm_dev.control.control_interface import control_step
from algorithm_dev.control.cooldown import get_cpu_temp, adaptive_cooldown

# import Tracking pipeline
from algorithm_dev.vision.track import Track #import Track class, TODO see if track needs to be passed/accessed
#from algorithm_dev.vision.tracking import process_video
#from algorithm_dev.control.laser_interface import LaserInterface
#from algorithm_dev.control.mirror_planner import MirrorPlanner
from algorithm_dev.Writer import Writer
from algorithm_dev.control.control_interface import init_hardware
from algorithm_dev.control.control_interface import init_hardware

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

    global mirror, laser , PAUSED

    print("[SYSTEM] Initializing hardware...")
    init_hardware() 

    # camera set up
    try:
        from algorithm_dev.vision.camera_interface import Camera
        cam = Camera()
        test_frame = cam.get_frame()
        if test_frame is None:
            raise RuntimeError("Camera returned no frames")
        print("[CAMERA] OK")
    except Exception as e:
        print("[CAMERA] FAIL:", e)
        cam = None

    # laser setup 
    try:
        laser = LaserInterface()  # initialize serial interface
        laser.off()  # ensure it starts off
        print("[LASER] OK")
    except Exception as e:
        print("[LASER] FAIL:", e)
        laser = None

    # mirror set up
    try:
        mirror = MirrorPlanner()  # initialize mirror
        mirror.off()  # ensure safe start
        print("[MIRROR] OK")
    except Exception as e:
        print("[MIRROR] FAIL:", e)
        mirror = None

    # hardware check
    if not all([cam, laser, mirror]):
        print("\n[SYSTEM] One or more subsystems failed to initialize. Exiting.")
        sys.exit(1)

    # signal handling
    signal.signal(signal.SIGINT, emergency_stop)      # CTRL+C (controlled shutdown)
    signal.signal(signal.SIGUSR1, toggle_pause)       # custom pause/resume; PID from ps aux | grep main.py, kill / resume -USR1 <PID>

    # writing set up
    writer = Writer()
  
    # === CALL VISION LOOP ===
    last_packet_time = time.perf_counter()
    WATCHDOG_TIMEOUT = 0.1 #100 ms, deadman watchdog
    
    try: 

        for packet in process_video(): 

            now = time.perf_counter()

            # watch dog, if vision stalls then turn off system
            if now - last_packet_time > WATCHDOG_TIMEOUT:
                print("[WATCHDOG] Vision timeout")
                if laser:
                    laser.off()
                if mirror: 
                    mirror.off()

            last_packet_time = now #system kills on lag 

            # pause handling 
            while PAUSED:
                if laser: 
                    laser.off()
                if mirror:
                    mirror.off()
                time.sleep(0.1)


            # CONTROL STEP 
            pipeline_start = packet["timestamp"] # starting time 
            result = control_step(packet["tracks"],
                                packet["states"],
                                packet["frame"]) 

            #CPU / thermal monitoring (from cooldown.py)
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
